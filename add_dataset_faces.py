import cv2 as cv
import os
import time

cam  = cv.VideoCapture(0)


def save_pics(faces_dir, amount_of_pics, name):
    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print('failed to grab a frame')
            break
        cv.imshow("test", frame)

        k = cv.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape was hit... closing")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = f"opencv_frame_{img_counter}"
            cv.imwrite(img_name, frame)
            print(f"{img_name} written")
            img_counter += 1

cam.release()

