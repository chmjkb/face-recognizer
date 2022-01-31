import cv2 as cv
import numpy as np
import os


def captureVid():
    """Function responsible for capturing the webcam"""
    while True:  # Reads the video frame by frame
        vid = cv.VideoCapture(0)

        ret, frame = vid.read()
        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break


haar_cascade = cv.CascadeClassifier('haar_cascades.xml')
people = []
DIR = os.path.join(os.getcwd(), 'faces')  # Faces dir location
for person in os.listdir(DIR)[1:]:  # Iterating through faces dir to get people
    people.append(person)

features = []
labels =[]


def create_training():
    """Grabbing every image in each folder and adding it to a training set"""
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)  # Gets the label of every single person in the faces dir

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in faces_rect:
                faces_roi = faces_rect[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)


