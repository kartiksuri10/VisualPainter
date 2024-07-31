import numpy as np
import cv2
import os
import HandsTrackingModule as htm
##################
brushThickness = 15
eraserThickness = 100
##################

folderPath = r"D:\VisualPainter\Header"
images = os.listdir(folderPath)

imageList = []
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
for imPath in images:
    image = cv2.imread(f'{folderPath}/{imPath}')
    imageList.append(image)

# print(len(imageList))

header = imageList[0]
drawColor = (0,0,255)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
xp, yp = 0, 0
detector = htm.handDetector(detectionCon=1)
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
while True:
    success, img = cap.read()
    if success:
        img = cv2.flip(img,1)

        img = detector.findHands(img)
        lmList = detector.findPositions(img, draw=False)
        if len(lmList)!=0:
            #index and middle finger coordinates
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            # find which fingers are up
            fingers = detector.fingersUp()
            
            # selection mode if both fingers are up
            if fingers[1] and fingers[2]:
                # print("Selection Mode")
                xp, yp = 0, 0
                if y1<125:
                    # print(x1, x2)
                    if 300<x1<450:
                        header = imageList[0]
                        drawColor = (0,0,255)
                    elif 600<x1<750:
                        header = imageList[1]
                        drawColor = (0,255,0)
                    elif 900<x1<1050:
                        header = imageList[2]
                        drawColor = (139,0,0)
                    elif 1100<x1<1280:
                        header = imageList[3]
                        drawColor = (0,0,0)
                cv2.rectangle(img, (x1,y1-25), (x2,y2+25), drawColor, cv2.FILLED)

            # drawing mode if index finger is up only
            if fingers[1] and fingers[2]==False:
                cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                # print("Drawing Mode")
                if drawColor == (0,0,0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

                xp, yp = x1, y1

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        img[0:125, 0:1280] = header
        cv2.imshow('Image', img)
        # cv2.imshow('Canvas', imgCanvas)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
