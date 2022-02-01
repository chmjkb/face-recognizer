import numpy as np
import cv2 as cv
import os


haar_cascade = cv.CascadeClassifier('haar_cascades.xml')
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('faces_trained.yml')

people = []
DIR = os.path.join(os.getcwd(), 'faces')  # Faces dir location
for person in os.listdir(DIR):  # Iterating through faces dir to get people
    people.append(person)


def capture_vid():
    """Function responsible for capturing the webcam"""

    vid = cv.VideoCapture(0)

    while True:  # Reads the video frame by frame

        ret, frame = vid.read()

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray_frame, 1.1, 4)  # Detecting a face

        for (x, y, w, h) in faces_rect:
            faces_roi = gray_frame[y:y+h, x:x+h]
            label, confidence = face_recognizer.predict(faces_roi)  # Recognizing a face
            print(f"Label = {label}, confidence = {confidence}")
            
            if confidence > 99:
                cv.putText(frame, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            else:
                cv.putText(frame, 'unknown', (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
                cv.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
            
        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

capture_vid()
