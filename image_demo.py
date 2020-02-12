# USAGE
# python image_demo.py --image sad.jpg --prototxt ./face_detector/deploy.prototxt.txt --model ./face_detector/res10_300x300_ssd_iter_140000.caffemodel

import os
import cv2
import tensorflow as tf
import pickle
import time
import argparse
import imutils
import numpy as np

import model
import data_utils
import data_provider

SMILE_INDEX = {0: 'Not Smile', 1: 'Smile'}
EMOTION_INDEX = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
GENDER_INDEX = {0: 'Female', 1: 'Male'}
AGE_INDEX = {0: '1-10', 1: '11-20', 2: '21-30', 3: '31-40', 4: '41-50', 5: '51-60', 6: '61-70'}

DEFAULT_CONFIDENCE = 0.5

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="image name")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print("OK !")

sess = tf.InteractiveSession()
test_model = model.Multitask_BKNet(sess, False)

def demo_image(image_name):
  image_path = os.path.join(os.getcwd(), image_name)
  image = cv2.imread(image_path)
  image = imutils.resize(image, width=600)
  (h, w) = image.shape[:2]

  imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                    (104.0, 177.0, 123.0), swapRB=False, crop=False)

  net.setInput(imageBlob)
  detections = net.forward()

  # loop over the detections
  for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > DEFAULT_CONFIDENCE:
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")

      # extract face ROI
      face = image[startY:endY, startX:endX]
      face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
      face = cv2.resize(face, (48, 48))
      face = (face-128.0) / 255
      T = np.zeros([48, 48, 1])
      T[:, :, 0] = face
      test_img = []
      test_img.append(T)
      test_img = np.asarray(test_img)

      feed_dict = {test_model.input_images: test_img, test_model.phase_train: False, test_model.keep_prob: 1}
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

    new_image_path = os.path.join(os.getcwd(), 'new_' + image_name.split('.')[0] + '.jpg')
    cv2.imwrite(new_image_path, image)

start = time.time()
demo_image(args["image"])
end = time.time()
print('Time: {}s'.format(end-start))
