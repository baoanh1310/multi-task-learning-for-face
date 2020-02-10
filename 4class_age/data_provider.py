import numpy as np
import cv2

import data_utils


class Dataset(object):
    def __init__(self, data_type, batch_size):
        self.data_type = data_type
        self.batch_size = batch_size

        self.read_data()
        self.convert_data_format()

    def get_one_hot_vector(self, num_classes, class_idx):
        '''
            Return tensor of shape (num_classes, )
        '''
        result = np.zeros(num_classes)
        result[class_idx] = 1.0

        return result

    def gen(self):
        np.random.shuffle(self.all_data)
        batch_images = []
        batch_labels = []
        batch_indexes = []

        for i in range(len(self.all_data)):
            if len(batch_images) == self.batch_size:
                batch_images = []
                batch_labels = []
                batch_indexes = []

            image, label, index = self.all_data[i]
            batch_images.append(image)
            batch_labels.append(label)
            batch_indexes.append(index)

            if len(batch_images) == self.batch_size:
                yield batch_images, batch_labels, batch_indexes
        if len(batch_images) > 0:
            yield batch_images, batch_labels, batch_indexes

    def read_data(self):
        self.smile_train, self.smile_test = data_utils.getSmileImage()
        self.emotion_train, self.emotion_public_test, self.emotion_private_test = data_utils.getEmotionImage()
        self.gender_train, self.gender_test = data_utils.getGenderImage()
        self.age_train, self.age_test = data_utils.getAgeImage()

    def convert_data_format(self):
        self.all_data = []

        if self.data_type == 'train':
            # Smile dataset
            for i in range(len(self.smile_train) * 10):
                image = (self.smile_train[i % 3000][0] - 128.0) / 255.0
                label = self.get_one_hot_vector(7, int(self.smile_train[i % 3000][1]))
                index = 1.0
                self.all_data.append((image, label, index))

            # Emotion dataset
            for i in range(len(self.emotion_train)):
                self.all_data.append((self.emotion_train[i][0], self.emotion_train[i][1], 2.0))

            # Gender dataset
            for i in range(len(self.gender_train)):
                image = (self.gender_train[i][0] - 128.0) / 255.0
                label = self.get_one_hot_vector(7, int(self.gender_train[i][1]))
                index = 3.0
                self.all_data.append((image, label, index))

            # Age dataset
            for i in range(len(self.age_train)):
                image = (self.age_train[i][0] - 128.0) / 255.0
                label = self.get_one_hot_vector(7, self.age_train[i][1])
                index = 4.0
                self.all_data.append((image, label, index))

        else:
            # Smile dataset
            for i in range(len(self.smile_test)):
                image = (self.smile_test[i][0] - 128.0) / 255.0
                label = self.get_one_hot_vector(7, int(self.smile_test[i][1]))
                index = 1.0
                self.all_data.append((image, label, index))

            # Gender dataset
            for i in range(len(self.gender_test)):
                image = (self.gender_test[i][0] - 128.0) / 255.0
                label = self.get_one_hot_vector(7, int(self.gender_test[i][1]))
                index = 3.0
                self.all_data.append((image, label, index))

            # Age dataset
            for i in range(len(self.age_test)):
                image = (self.age_test[i][0] - 128.0) / 255.0
                label = self.get_one_hot_vector(7, self.age_test[i][1])
                index = 4.0
                self.all_data.append((image, label, index))

            if self.data_type == 'public_test':
                for i in range(len(self.emotion_public_test)):
                    self.all_data.append((self.emotion_public_test[i][0], self.emotion_public_test[i][1], 2.0))
            else:
                for i in range(len(self.emotion_private_test)):
                    self.all_data.append((self.emotion_private_test[i][0], self.emotion_private_test[i][1], 2.0))


# d = Dataset(data_type='train', batch_size=32)
