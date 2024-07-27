import numpy as np
import cv2
import os
import time

folderPath = r"D:\VisualPainter\Header"
images = os.listdir(folderPath)

imageList = []

for imPath in images:
    image = cv2.imread(f'{folderPath}/{imPath}')
    imageList.append(image)

print(len(imageList))

header = imageList[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    if success:
        img = cv2.flip(img,1)
        img[0:125, 0:1280] = header
        cv2.imshow('Image', img)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
