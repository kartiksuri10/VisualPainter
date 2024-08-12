import numpy as np
import cv2
import os
import HandsTrackingModule as htm
from scipy.spatial import distance
##################
brushThickness = 10
eraserThickness = 100
##################
def recognize_shape(points):
    # Analyze the points and recognize the shape
    if len(points) < 5:
        return "line"

    # Find bounding box
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    # Check if it is a rectangle
    if is_rectangle(points, bbox):
        return "rectangle"
    elif is_circle(points, bbox):
        return "circle"
    elif is_triangle(points, bbox):
            return "triangle"
    return "unknown"

def draw_shape(canvas, points, shape, color, thickness, bbox):
    x_min, y_min, x_max, y_max = bbox
    if shape == "line":
        cv2.line(canvas, points[0], points[-1], color, thickness)
    elif shape == "rectangle":
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        canvas[y_min:y_max, x_min:x_max] = 0
        cv2.rectangle(canvas, (min(x_coords), min(y_coords)), (max(x_coords), max(y_coords)), color, thickness)
    elif shape == "circle":
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        center = (sum(x_coords) // len(x_coords), sum(y_coords) // len(y_coords))
        radius = int(np.mean([distance.euclidean(center, p) for p in points]))
        canvas[y_min-15:y_max+15, x_min-15:x_max+15] = 0
        cv2.circle(canvas, center, radius, color, thickness)

    elif shape == "triangle":
        # Convert points to a numpy array
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        epsilon = 0.02* cv2.arcLength(pts, True)
        approx = cv2.approxPolyDP(pts, epsilon, True)
        canvas[y_min-15:y_max+15, x_min-15:x_max+15] = 0
        cv2.polylines(canvas, [approx], isClosed=True, color=color, thickness=thickness)


def is_rectangle(points, bbox):
    x_min, y_min, x_max, y_max = bbox
    threshold = 15  # Allowable distance from the bounding box

    # Check how many points are near the bounding box edges
    near_edges = 0
    for (x, y) in points:
        if (abs(x - x_min) < threshold or abs(x - x_max) < threshold or
            abs(y - y_min) < threshold or abs(y - y_max) < threshold):
            near_edges += 1

    # Consider it a rectangle if most points are near the edges
    if near_edges > 0.85 * len(points):
        return True
    return False
def is_circle(points, bbox):
    x_min, y_min, x_max, y_max = bbox
    center = ((x_min+x_max)//2,(y_min+y_max)//2)
    distances = [distance.euclidean(center, (x, y)) for x, y in points]
    mean_radius = np.mean(distances)
    radius_threshold = 20  # Allowable variation in radius
    for dist in distances:
        if abs(dist - mean_radius) > radius_threshold:
            return False
    return True

def is_triangle(points, bbox):
    # Convert the points to a numpy array
    pts = np.array(points, dtype=np.int32)
    
    # Reshape the points for the contour function
    pts = pts.reshape((-1, 1, 2))
    
    # Approximate the contour to a polygon
    epsilon = 0.05 * cv2.arcLength(pts, True)
    approx = cv2.approxPolyDP(pts, epsilon, True)
    
    # Check if the approximation has 3 vertices (triangle)
    if len(approx) == 3:
        return True
    
    return False

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
detector = htm.handDetector(detectionCon=0.85)

# window resizeable
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

# Store points for shape recognition
points = []

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
                    if 300<x1<400:
                        header = imageList[0]
                        drawColor = (0,0,255)
                    elif 550<x1<700:
                        header = imageList[1]
                        drawColor = (0,255,0)
                    elif 850<x1<1050:
                        header = imageList[2]
                        drawColor = (224, 162, 2)
                    elif 1100<x1<1280:
                        header = imageList[3]
                        drawColor = (0,0,0)
                # Recognize shape
                if len(points) > 10:  # Minimum points for a shape
                    recognized_shape = recognize_shape(points)

                    bbox = (min([p[0] for p in points]), min([p[1] for p in points]),
                            max([p[0] for p in points]), max([p[1] for p in points]))
                    
                    draw_shape(imgCanvas, points, recognized_shape, drawColor, brushThickness, bbox)
                points = []  # Clear points after recognizing shape
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
                points.append((x1, y1))
                xp, yp = x1, y1

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        img[0:125, 0:1280] = header
        cv2.imshow('Image', img)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
            break
    else:
        break
    

cap.release()
cv2.destroyAllWindows()
