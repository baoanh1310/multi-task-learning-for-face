import tensorflow as tf
import numpy as np
import os

import config
import data_provider
import data_utils

class Multitask_BKNet(object):
    def __init__(self, session, is_training=True):
        self.global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
        self.model_dir = config.MODEL_DIR
        self.num_epochs = config.NUM_EPOCHS
        self.batch_size = config.BATCH_SIZE
        self.sess = session

        # Building model
        self._define_input()
        self._build_model()
        self._define_loss()

        # Extra variables
        smile_correct_prediction = tf.equal(tf.argmax(self.y_smile_conv, 1), tf.argmax(self.y_smile, 1))
        emotion_correct_prediction = tf.equal(tf.argmax(self.y_emotion_conv, 1), tf.argmax(self.y_emotion, 1))
        gender_correct_prediction = tf.equal(tf.argmax(self.y_gender_conv, 1), tf.argmax(self.y_gender, 1))
        age_correct_prediction = tf.equal(tf.argmax(self.y_age_conv, 1), tf.argmax(self.y_age, 1))

        self.smile_true_pred = tf.reduce_sum(tf.cast(smile_correct_prediction, dtype=tf.float32) * self.smile_mask)
        self.emotion_true_pred = tf.reduce_sum(tf.cast(emotion_correct_prediction, dtype=tf.float32) * self.emotion_mask)
        self.gender_true_pred = tf.reduce_sum(tf.cast(gender_correct_prediction, dtype=tf.float32) * self.gender_mask)
        self.age_true_pred = tf.reduce_sum(tf.cast(age_correct_prediction, dtype=tf.float32) * self.age_mask)


        # Learning rate and train op
        self.train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.total_loss, global_step=self.global_step)
        # self.train_step = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=config.INIT_LR).minimize(self.total_loss, global_step=self.global_step)

        if is_training:
            self.train_data = data_provider.Dataset('train', self.batch_size)
        # self.public_test_data = data_provider.Dataset('public_test', self.batch_size)
        # self.private_test_data = data_provider.Dataset('private_test', self.batch_size)

        self.saver_all = tf.train.Saver(tf.all_variables(), max_to_keep=5)
        self.checkpoint_path = os.path.join(self.model_dir, "model.ckpt")
        ckpt = tf.train.get_checkpoint_state(os.path.join(os.getcwd(), config.MODEL_DIR))
        if ckpt:
            print("Reading model parameters from %s", ckpt.model_checkpoint_path)
            self.saver_all.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.initialize_all_variables())

    def _define_input(self):
        self.input_images = tf.placeholder(tf.float32, [None, config.IMAGE_SIZE, config.IMAGE_SIZE, 1])
        self.input_labels = tf.placeholder(tf.float32, [None, 7])
        self.input_indexes = tf.placeholder(tf.float32, [None])

        self.phase_train = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            filter = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                                    tf.truncated_normal_initializer(stddev=config.WEIGHT_INIT))
            return tf.nn.conv2d(x, filter, [1, strides, strides, 1], 'SAME')

    def _relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _batch_norm(self, x, n_out, phase_train=True, scope='bn'):
        with tf.variable_scope(scope):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def _FC(self, name, x, out_dim, keep_rate, activation='relu'):
        assert (activation == 'relu') or (activation == 'softmax') or (activation == 'linear')
        with tf.variable_scope(name):
            dim = x.get_shape().as_list()
            dim = np.prod(dim[1:])
            x = tf.reshape(x, [-1, dim])
            W = tf.get_variable('DW', [x.get_shape()[1], out_dim],
                                initializer=tf.truncated_normal_initializer(stddev=config.WEIGHT_INIT))
            b = tf.get_variable('bias', [out_dim], initializer=tf.constant_initializer())
            x = tf.nn.xw_plus_b(x, W, b) # linear
            if activation == 'relu':
                x = self._relu(x)
            else:
                if activation == 'softmax':
                    x = tf.nn.softmax(x)

            if activation != 'relu':
                return x
            else:
                return tf.nn.dropout(x, keep_rate)


    def _max_pool(self, x, filter, stride):
        return tf.nn.max_pool(x, [1, filter, filter, 1], [1, stride, stride, 1], 'SAME')

    def VGG_ConvBlock(self, name, x, in_filters, out_filters, repeat, strides, phase_train):
        with tf.variable_scope(name):
            for layer in range(repeat):
                scope_name = name + '_' + str(layer)
                x = self._conv(scope_name, x, 3, in_filters, out_filters, strides)
                if config.USE_BN:
                    x = self._batch_norm(x, out_filters, phase_train)
                x = self._relu(x)

                in_filters = out_filters

            x = self._max_pool(x, 2, 2)
            return x

    def _build_model(self):
        x = self.input_images

        x = self.VGG_ConvBlock('Block1', x, 1, 32, 2, 1, self.phase_train)
        print(x.get_shape())

        x = self.VGG_ConvBlock('Block2', x, 32, 64, 2, 1, self.phase_train)
        print(x.get_shape())

        x = self.VGG_ConvBlock('Block3', x, 64, 128, 2, 1, self.phase_train)
        print(x.get_shape())

        x = self.VGG_ConvBlock('Block4', x, 128, 256, 3, 1, self.phase_train)
        print(x.get_shape())

        # Smile branch
        smile_fc1 = self._FC('smile_fc1', x, 256, self.keep_prob)
        smile_fc2 = self._FC('smile_fc2', smile_fc1, 256, self.keep_prob)
        # self.y_smile_conv = self._FC('smile_softmax', smile_fc2, 2, self.keep_prob, 'linear')
        self.y_smile_conv = self._FC('smile_softmax', smile_fc2, 2, self.keep_prob, 'softmax')

        # Emotion branch
        emotion_fc1 = self._FC('emotion_fc1', x, 256, self.keep_prob)
        emotion_fc2 = self._FC('emotion_fc2', emotion_fc1, 256, self.keep_prob)
        # self.y_emotion_conv = self._FC('emotion_softmax', emotion_fc2, 7, self.keep_prob, 'linear')
        self.y_emotion_conv = self._FC('emotion_softmax', emotion_fc2, 7, self.keep_prob, 'softmax')

        # Gender branch
        gender_fc1 = self._FC('gender_fc1', x, 256, self.keep_prob)
        gender_fc2 = self._FC('gender_fc2', gender_fc1, 256, self.keep_prob)
        # self.y_gender_conv = self._FC('gender_softmax', gender_fc2, 2, self.keep_prob, 'linear')
        self.y_gender_conv = self._FC('gender_softmax', gender_fc2, 2, self.keep_prob, 'softmax')

        # Age branch
        age_fc1 = self._FC('age_fc1', x, 256, self.keep_prob)
        age_fc2 = self._FC('age_fc2', age_fc1, 256, self.keep_prob)
        # self.y_age_conv = self._FC('age_softmax', age_fc2, 7, self.keep_prob, 'linear')
        # self.y_age_conv = self._FC('age_softmax', age_fc2, 4, self.keep_prob, 'linear')
        self.y_age_conv = self._FC('age_softmax', age_fc2, 4, self.keep_prob, 'softmax')

    def _define_loss(self):
        self.smile_mask = tf.cast(tf.equal(self.input_indexes, 1), tf.float32)
        self.emotion_mask = tf.cast(tf.equal(self.input_indexes, 2), tf.float32)
        self.gender_mask = tf.cast(tf.equal(self.input_indexes, 3), tf.float32)
        self.age_mask = tf.cast(tf.equal(self.input_indexes, 4), tf.float32)

        self.y_smile = self.input_labels[:, :2]
        self.y_emotion = self.input_labels[:, :7]
        self.y_gender = self.input_labels[:, :2]
        self.y_age = self.input_labels[:, :4]

        self.smile_cross_entropy = tf.reduce_sum(
            tf.reduce_sum(-self.y_smile * tf.log(tf.clip_by_value(tf.nn.softmax(self.y_smile_conv), 1e-10, 1.0)),
                        axis=1) * self.smile_mask) / tf.clip_by_value(tf.reduce_sum(self.smile_mask), 1, int(1e9))

        self.emotion_cross_entropy = tf.reduce_sum(
            tf.reduce_sum(-self.y_emotion * tf.log(tf.clip_by_value(tf.nn.softmax(self.y_emotion_conv), 1e-10, 1.0)),
                        axis=1) * self.emotion_mask) / tf.clip_by_value(tf.reduce_sum(self.emotion_mask), 1, int(1e9))

        self.gender_cross_entropy = tf.reduce_sum(
            tf.reduce_sum(-self.y_gender * tf.log(tf.clip_by_value(tf.nn.softmax(self.y_gender_conv), 1e-10, 1.0)),
                        axis=1) * self.gender_mask) / tf.clip_by_value(tf.reduce_sum(self.gender_mask), 1, int(1e9))

        self.age_cross_entropy = tf.reduce_sum(
            tf.reduce_sum(-self.y_age * tf.log(tf.clip_by_value(tf.nn.softmax(self.y_age_conv), 1e-10, 1.0)),
                        axis=1) * self.age_mask) / tf.clip_by_value(tf.reduce_sum(self.age_mask), 1, int(1e9))

        self.emotion_hinge_loss = tf.maximum(self.y_emotion_conv - tf.reduce_sum(self.y_emotion_conv * self.y_emotion, axis=1, keep_dims=True) + 1.0, 0.0)
        self.emotion_hinge_loss = tf.reduce_sum(tf.reduce_sum(self.emotion_hinge_loss * self.emotion_hinge_loss, axis=1) * self.emotion_mask) / tf.reduce_sum(self.emotion_mask)

        self.age_hinge_loss = tf.maximum(self.y_age_conv - tf.reduce_sum(self.y_age_conv * self.y_age, axis=1, keep_dims=True) + 1.0, 0.0)
        self.age_hinge_loss = tf.reduce_sum(tf.reduce_sum(self.age_hinge_loss * self.age_hinge_loss, axis=1) * self.age_mask) / tf.reduce_sum(self.age_mask)

        l2_loss = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                l2_loss.append(tf.nn.l2_loss(var))
        self.l2_loss = config.WEIGHT_DECAY * tf.add_n(l2_loss)

        self.total_loss = self.smile_cross_entropy + self.emotion_cross_entropy + self.gender_cross_entropy + self.age_cross_entropy + self.l2_loss
        # self.total_loss = self.smile_cross_entropy + self.emotion_hinge_loss + self.gender_cross_entropy + self.age_hinge_loss + self.l2_loss

    def train(self):
        current_step = self.sess.run(self.global_step)

        for epoch in range(self.num_epochs):
            avg_ttl = []
            avg_rgl = []
            avg_smile_loss = []
            avg_emotion_loss = []
            avg_gender_loss = []
            avg_age_loss = []

            smile_nb_true_pred = 0
            emotion_nb_true_pred = 0
            gender_nb_true_pred = 0
            age_nb_true_pred = 0

            smile_nb_train = 0
            emotion_nb_train = 0
            gender_nb_train = 0
            age_nb_train = 0
            print('Epoch: ', epoch)
            for batch_image, batch_label, batch_index in self.train_data.gen():
                for i in range(len(batch_index)):
                    if batch_index[i] == 1.0:
                        smile_nb_train += 1
                    elif batch_index[i] == 2.0:
                        emotion_nb_train += 1
                    elif batch_index[i] == 3.0:
                        gender_nb_train += 1
                    else:
                        age_nb_train += 1

                batch_image = data_utils.augmentation(batch_image, 48)

                feed_dict = {self.input_images: batch_image, \
                            self.input_labels: batch_label, \
                            self.input_indexes: batch_index, \
                            self.phase_train: True, \
                            self.keep_prob: 0.5}

                ttl, sml, eml, gel, agel, l2l, _ = self.sess.run(\
                    [self.total_loss, self.smile_cross_entropy, self.emotion_cross_entropy, self.gender_cross_entropy, self.age_cross_entropy,\
                        self.l2_loss, self.train_step], feed_dict=feed_dict)

                smile_nb_true_pred += self.sess.run(self.smile_true_pred, feed_dict=feed_dict)

                emotion_nb_true_pred += self.sess.run(self.emotion_true_pred,
                                                feed_dict=feed_dict)

                gender_nb_true_pred += self.sess.run(self.gender_true_pred,
                                                feed_dict=feed_dict)

                age_nb_true_pred += self.sess.run(self.age_true_pred,
                                                feed_dict=feed_dict)

                print('smile_loss: %.2f, emotion_loss: %.2f, gender_loss: %.2f, age_loss: %.2f, l2_loss: %.2f, total_loss: %.2f\r'%(\
                    sml, eml, gel, agel, l2l, ttl), end="")

                avg_ttl.append(ttl)
                avg_smile_loss.append(sml)
                avg_emotion_loss.append(eml)
                avg_gender_loss.append(gel)
                avg_age_loss.append(agel)
                avg_rgl.append(l2l)

            smile_train_accuracy = smile_nb_true_pred * 1.0 / smile_nb_train
            emotion_train_accuracy = emotion_nb_true_pred * 1.0 / emotion_nb_train
            gender_train_accuracy = gender_nb_true_pred * 1.0 / gender_nb_train
            age_train_accuracy = age_nb_true_pred * 1.0 / age_nb_train

            avg_smile_loss = np.average(avg_smile_loss)
            avg_emotion_loss = np.average(avg_emotion_loss)
            avg_gender_loss = np.average(avg_gender_loss)
            avg_age_loss = np.average(avg_age_loss)
            avg_rgl = np.average(avg_rgl)
            avg_ttl = np.average(avg_ttl)

            print('\n')

            print('Smile task train accuracy: ' + str(smile_train_accuracy * 100))
            print('Emotion task train accuracy: ' + str(emotion_train_accuracy * 100))
            print('Gender task train accuracy: ' + str(gender_train_accuracy * 100))
            print('Age task train accuracy: ' + str(age_train_accuracy * 100))
            print('Total loss: ' + str(avg_ttl) + '. L2-loss: ' + str(avg_rgl))
            print('Smile loss: ' + str(avg_smile_loss))
            print('Emotion loss: ' + str(avg_emotion_loss))
            print('Gender loss: ' + str(avg_gender_loss))
            print('Age loss: ' + str(avg_age_loss))

            self.saver_all.save(self.sess, self.model_dir + '/model.ckpt')

    def count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))
