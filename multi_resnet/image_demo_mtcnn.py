# USAGE
# python image_demo_mtcnn.py --image ../surprise.jpg

import os
import cv2
import tensorflow as tf
import pickle
import time
import argparse
import imutils
import numpy as np
from mtcnn.mtcnn import MTCNN

import model
import data_utils
import data_provider

SMILE_INDEX = {0: 'Not Smile', 1: 'Smile'}
EMOTION_INDEX = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
GENDER_INDEX = {0: 'Female', 1: 'Male'}
AGE_INDEX = {0: 'Too Young', 1: 'Young', 2: 'Old', 3: 'Too Old'}

DEFAULT_CONFIDENCE = 0.5

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="image name")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load face detector
print("[INFO] loading face detector...")
detector = MTCNN()
print("OK!")

sess = tf.InteractiveSession()
print("[INFO] loading ResNet model...")
test_model = model.ResNet_v1(sess, False)
print("OK!")

def demo_image(image_name):
    image_path = os.path.join(os.getcwd(), image_name)
    image = cv2.imread(image_path)
    # image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    faces = detector.detect_faces(image)
    for face in faces:
        confidence = face['confidence']

        if confidence > DEFAULT_CONFIDENCE:
            x, y, width, height = face['box']
            startX = int(x)
            startY = int(y)
            endX = int(startX + width)
            endY = int(startY + height)

            # extract face ROI
            img_face = image[startY:endY, startX:endX]
            img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
            img_face = cv2.resize(img_face, (48, 48))
            img_face = (img_face - 128.0) / 255
            T = np.zeros([48, 48, 1])
            T[:, :, 0] = img_face
            test_img = []
            test_img.append(T)
            test_img = np.asarray(test_img)

            feed_dict = {test_model.input_images: test_img, test_model.is_training: False, test_model.keep_prob: 1.0}
            predict_y_smile_conv = np.argmax(sess.run(test_model.y_smile_conv, feed_dict=feed_dict))
            predict_y_emotion_conv = np.argmax(sess.run(test_model.y_emotion_conv, feed_dict=feed_dict))
            predict_y_gender_conv = np.argmax(sess.run(test_model.y_gender_conv, feed_dict=feed_dict))
            predict_y_age_conv = np.argmax(sess.run(test_model.y_age_conv, feed_dict=feed_dict))

            smile_label = SMILE_INDEX[predict_y_smile_conv]
            emotion_label = EMOTION_INDEX[predict_y_emotion_conv]
            gender_label = GENDER_INDEX[predict_y_gender_conv]
            age_label = AGE_INDEX[predict_y_age_conv]

            text = "{}, {}, {}, {}".format(smile_label, emotion_label, gender_label, age_label)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    name = image_name.split('/')[1].split('.')[0]
    new_image_path = os.path.join(os.getcwd(), 'new_' + name + '.jpg')
    cv2.imwrite(new_image_path, image)

demo_image(args["image"])
