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
        
    vid.release()
    cv.destroyAllWindows()


people = [f.path.lstrip('\\faces') for f in os.scandir('faces') if f.is_dir()]
faces_dir = os.getcwd() + '\\faces'
haar_cascade = cv.CascadeClassifier('haar_cascades.xml')


def create_faces(directory, people_list):
    features = []
    labels = []

    for person in people_list:
        path = os.path.join(directory, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)


def train_model(features, labels):
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    features = np.array(features, dtype='object')
    labels = np.array(labels)

    np.save('features.npy', features)
    np.save('labels.npy', labels)

    # Train the recognizer on the features and labels list
    face_recognizer.train(features, labels)


create_faces(faces_dir, people)