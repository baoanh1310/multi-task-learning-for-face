import numpy as np
import os
import cv2
import scipy
import random

SMILE_FOLDER = './data/smile_data/'
EMOTION_FOLDER = './data/emotion_data2/'
GENDER_FOLDER = './data/wiki_data_gender/'
AGE_FOLDER = './data/age_data/'
IMDB_FOLDER = './data/imdb_data/'
NUM_SMILE_IMAGE = 4000
SMILE_SIZE = 48
EMOTION_SIZE = 48

def getAgeImage():
    print('Load age image..................')
    # X1 = np.load(AGE_FOLDER + 'train_48_crop.npy', allow_pickle=True)
    # X2 = np.load(AGE_FOLDER + 'test_48_crop.npy', allow_pickle=True)
    X1 = np.load(AGE_FOLDER + 'train_48_crop_4class.npy', allow_pickle=True)
    X2 = np.load(AGE_FOLDER + 'test_48_crop_4class.npy', allow_pickle=True)
    X3 = np.load(AGE_FOLDER + 'train_utk.npy', allow_pickle=True)

    train_data = []
    test_data = []

    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X3.shape[0]):
        train_data.append(X3[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Done !')
    print('Number of age train data: ', str(len(train_data)))
    print('Number of age test data: ', str(len(test_data)))
    print('---------------------------------------------------------------')
    return train_data, test_data

def getSmileImage():
    print('Load smile image...................')
    X1 = np.load(SMILE_FOLDER + 'train.npy', allow_pickle=True)
    X2 = np.load(SMILE_FOLDER + 'test.npy', allow_pickle=True)

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Done !')
    print('Number of smile train data: ',str(len(train_data)))
    print('---------------------------------------------------------------')
    return train_data, test_data


def getGenderImage():
    print('Load gender image...................')
    X1 = np.load(GENDER_FOLDER + 'train.npy', allow_pickle=True)
    X2 = np.load(GENDER_FOLDER + 'test.npy', allow_pickle=True)

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])


    print('Done !')
    print('Number of gender train data: ', str(len(train_data)))
    print('---------------------------------------------------------------')
    return train_data, test_data


# def getEmotionImage():
#     print('Load emotion image..................')
#     train_images, train_labels, validation_images, validation_labels = [], [], [], []
#     if os.path.isfile(EMOTION_FOLDER + 'temp_train_and_validation/train/images.npy'):
#         train_images = np.load(EMOTION_FOLDER + 'temp_train_and_validation/train/images.npy', allow_pickle=True)
#         train_labels = np.load(EMOTION_FOLDER + 'temp_train_and_validation/train/labels.npy', allow_pickle=True)
#         validation_images = np.load(EMOTION_FOLDER + 'temp_train_and_validation/validation/images.npy', allow_pickle=True)
#         validation_labels = np.load(EMOTION_FOLDER + 'temp_train_and_validation/validation/labels.npy', allow_pickle=True)
#     train_data = []
#     for i in range(len(train_images)):
#         T = train_images[i]
#         label = train_labels[i]
#         train_data.append((T, label))
#
#     for i in range(len(validation_images)):
#         T = validation_images[i]
#         label = validation_labels[i]
#         train_data.append((T, label))
#     np.random.shuffle(train_data)
#
#     public_test_images = np.load(EMOTION_FOLDER + 'public test/images.npy', allow_pickle=True)
#     public_test_labels = np.load(EMOTION_FOLDER + 'public test/labels.npy', allow_pickle=True)
#     private_test_images = np.load(EMOTION_FOLDER + 'private test/images.npy', allow_pickle=True)
#     private_test_labels = np.load(EMOTION_FOLDER + 'private test/labels.npy', allow_pickle=True)
#
#     public_test_data = []
#     private_test_data = []
#     for i in range(len(public_test_images)):
#         T = public_test_images[i]
#         label = public_test_labels[i]
#         public_test_data.append((T, label))
#
#     for i in range(len(private_test_images)):
#         T = private_test_images[i]
#         label = private_test_labels[i]
#         private_test_data.append((T, label))
#
#     print('Done !')
#     print('Number of emotion train data: ', str(len(train_data)))
#     print('---------------------------------------------------------------')
#     return train_data, public_test_data, private_test_data

def getEmotionImage():
    print("Load emotion images.....................")
    train_data = np.load(EMOTION_FOLDER + 'train.npy', allow_pickle=True, encoding='latin1')
    public_test_data = np.load(EMOTION_FOLDER + 'public_test.npy', allow_pickle=True, encoding='latin1')
    private_test_data = np.load(EMOTION_FOLDER + 'private_test.npy', allow_pickle=True, encoding='latin1')

    print('Done!')
    print('Number of emotion train data: ', str(len(train_data)))
    print('-----------------------------------------------------------------')
    return train_data, public_test_data, private_test_data


''' Data augmentation method '''


def random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])
    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad, mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    return new_batch


def random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def random_flip_updown(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.flipud(batch[i])
    return batch


def random_90degrees_rotation(batch, rotations=[0, 1, 2, 3]):
    for i in range(len(batch)):
        num_rotations = random.choice(rotations)
        batch[i] = np.rot90(batch[i], num_rotations)
    return batch


def random_rotation(batch, max_angle):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            angle = random.uniform(-max_angle, max_angle)
            batch[i] = scipy.ndimage.interpolation.rotate(batch[i], angle, reshape=False)
    return batch


def random_blur(batch, sigma_max=5.0):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            sigma = random.uniform(0., sigma_max)
            batch[i] = scipy.ndimage.filters.gaussian_filter(batch[i], sigma)
    return batch


def augmentation(batch, img_size):
    batch = random_crop(batch, (img_size, img_size), 10)
    #batch = random_blur(batch)
    batch = random_flip_leftright(batch)
    batch = random_rotation(batch, 15)

    return batch
