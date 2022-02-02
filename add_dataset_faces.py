"""import cv2 as cv
import os
import time


def save_pics(faces_dir, amount_of_pics, name):
    while True:
        vid = cv.VideoCapture(0)
        ret, frame = vid.read()
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        for counter in range(amount_of_pics):
            pic_path = os.path.join(os.path.join(faces_dir, name), str(counter) + '.jpg')
            cv.imwrite(pic_path, gray_frame)


#name = input('')
DIR = os.path.join(os.getcwd(), 'faces')  # Faces dir location
save_pics(DIR, 10, 'Kacper Kaczmarczyk')

"""