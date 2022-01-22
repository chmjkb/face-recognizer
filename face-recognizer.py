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
    
people = [ f.path.lstrip('\\Faces') for f in os.scandir('Faces') if f.is_dir() ]
faces_dir = os.getcwd() + '\\Faces'


def trainModel(DIR, people):
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

    
    



