import cv2 as cv
import numpy as np
import os


def get_people_dir(DIR):
    people = []
    for person in os.listdir(DIR):  # Iterating through faces dir to get people
        if not person.startswith('.'):  # Prevents adding the hidden files to the people list
            people.append(person)

    return people


def get_features_labels(DIR, people):
    """Grabbing every image in each folder and adding it to a training set"""
    haar_cascade = cv.CascadeClassifier('haar_cascades.xml')

    features = []
    labels = []

    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)  # Gets the label of every single person in the faces dir

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            if img_array is None:
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(20, 20)
            )
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
    return features, labels


def train_model(features, labels):
    """Training the model and saving them to yml file"""
    features = np.array(features)
    labels = np.array(labels)

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(features, labels)

    face_recognizer.save('faces_trained.yml')
    np.save('features.npy', features)
    np.save('labels.npy', labels)


def facemask_training():
    """Face mask detection training"""
    pass

