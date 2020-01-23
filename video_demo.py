# USAGE
# python video_demo.py --video clip.mp4 --prototxt ./face_detector/deploy.prototxt.txt --model ./face_detector/res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from imutils.video import FileVideoStream
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import tensorflow as tf
import model

# load multi-task model
sess = tf.InteractiveSession()
print("[INFO] loading multitask model...")
test_model = model.Multitask_BKNet(sess, False)
print("OK!")

SMILE_INDEX = {0: 'Not Smile', 1: 'Smile'}
EMOTION_INDEX = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
GENDER_INDEX = {0: 'Female', 1: 'Male'}
AGE_INDEX = {0: '1-10', 1: '11-20', 2: '21-30', 3: '31-40', 4: '41-50', 5: '51-60', 6: '61-70'}


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
    help="input video name")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading face detector...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print("OK!")

# init video
print("[INFO] starting video file...")
fvs = FileVideoStream(args["video"]).start()
time.sleep(2.0)

while fvs.more():
    frame = fvs.read()
    frame = imutils.resize(frame, width=800)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < args["confidence"]:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        img = frame[startY:endY, startX:endX]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))
        img = (img-128.0) / 255.0
        T = np.zeros([48, 48, 1])
        T[:, :, 0] = img
        test_img = []
        test_img.append(T)
        test_img = np.asarray(test_img)

        feed_dict = {test_model.input_images: test_img, test_model.phase_train:False, test_model.keep_prob:1.0}
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
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
