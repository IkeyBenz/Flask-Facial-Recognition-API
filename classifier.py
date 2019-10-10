import os

import cv2
import numpy as np

clf = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray_img, scaleFactor=1.08, minNeighbors=5)

    return img, faces

def draw_face_rectangles(img, faces):
    """Adds rectangles to the given img in every x,y,w,h coordinate in faces"""
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

def api_controller(args):
    args.file.save('tmp_img.jpg')
    img, faces = detect_faces('tmp_img.jpg')
    draw_face_rectangles(img, faces)
    cv2.imwrite('with_faces.jpg', img)

    return 'with_faces.jpg'


if __name__ == '__main__':
    img, faces = detect_faces('test2.jpg')
    draw_face_rectangles(img, faces)
    cv2.imwrite('with_faces.jpg', img)
