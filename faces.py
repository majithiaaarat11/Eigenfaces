import os
import sys
import cv2
import numpy as np

cascadePath = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascadePath)

def read_images(path):
    image_paths = []
    folder_paths = []

    for folderPath in sorted(os.listdir(path)):
                for filePath in os.listdir(os.path.join(path,folderPath)):
                        fileExt = os.path.splitext(filePath)[1]
                        if fileExt in [".jpg", ".jpeg"]:

                                # Add to array of images
                                imagePath = os.path.join(path, folderPath, filePath)
                                image_paths.append(imagePath)

    for image_path in image_paths:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (660,510), interpolation = cv2.INTER_AREA)
        img = cv2.rectangle(img, (200,150),(450,360),(255,0,0),2)
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


"""
    for image_path in image_paths:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (640,500), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]


        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
"""


read_images('data')




#[1008:2016,1344:2688]