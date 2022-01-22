from re import L
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

def getImages():
    pass
