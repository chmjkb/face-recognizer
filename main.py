import numpy as np
import cv2 as cv


haar_cascade = cv.CascadeClassifier('haar_cascades.xml')
features = np.load('features.npy')
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('faces_trained.yml')


def captureVid():
    """Function responsible for capturing the webcam"""
    while True:  # Reads the video frame by frame
        vid = cv.VideoCapture(0)

        ret, frame = vid.read()
        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break


