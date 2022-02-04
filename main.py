import numpy as np
import cv2 as cv
import os
from add_dataset_faces import save_pics
from face_recognizer import get_people_dir

haar_cascade = cv.CascadeClassifier('haar_cascades.xml')
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('faces_trained.yml')


def capture_vid(user_confidence):
    """Function responsible for capturing the webcam"""
    DIR = os.path.join(os.getcwd(), 'faces')  # Faces dir location
    vid = cv.VideoCapture(0)
    while True:  # Reads the video frame by frame

        ret, frame = vid.read()
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        cv.putText(frame, "Press spacebar to take a photo", (20, 20), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)

        faces_rect = haar_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(20, 20)
            )  # Detecting a face

        k = cv.waitKey(1)
        if k % 256 == 32:
            save_pics(DIR, frame)

        if k & 0xFF == ord('q'):  # Press 'q' to quit
            break

        people = get_people_dir(DIR)
        for (x, y, w, h) in faces_rect:
            faces_roi = gray_frame[y:y+h, x:x+h]
            label, confidence = face_recognizer.predict(faces_roi)  # Recognizing a face
            #print(f"Label = {label}, confidence = {confidence}")

            if confidence > user_confidence:
                cv.putText(frame, str(people[label]), (x+20, y+20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 1)
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv.putText(frame, 'unknown', (x+20, y+20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        cv.imshow('frame', frame)


while True:
    try:
        user_confidence = int(input("Select confidence:"))
        break
    except ValueError:
        print("Enter a number!")

capture_vid(user_confidence)
