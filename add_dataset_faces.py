import cv2 as cv
import os
from datetime import datetime


def save_pics(faces_dir, frame):
    """Function responsible for saving the picture once spacebar is pressed"""

    name = input('What is your name?')  # We need that to define where to save the picture
    img_name = f"{datetime.now()}.jpg"
    path = os.path.join(faces_dir, name).replace(" ", "\\ ")
    print(f"img saved to {path}")
    print(os.path.join(path, img_name))

    cv.imwrite(os.path.join(path, img_name), frame)



