import cv2 as cv
import numpy as np
import os


haar_cascade = cv.CascadeClassifier('haar_cascades.xml')
people = []
DIR = os.path.join(os.getcwd(), 'faces')  # Faces dir location
for person in os.listdir(DIR):  # Iterating through faces dir to get people
    people.append(person)

features = []
labels = []


def create_training():
    """Grabbing every image in each folder and adding it to a training set"""
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)  # Gets the label of every single person in the faces dir

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            if img_array is None:
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)


create_training()
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Training the recognizer
face_recognizer.train(features, labels)

face_recognizer.save('faces_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)




