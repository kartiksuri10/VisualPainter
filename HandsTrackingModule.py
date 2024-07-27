import mediapipe as mp
import cv2

#open webcam
cap = cv2.VideoCapture(0)
# solutions are the submodules provided by mediapipe (Face detection, Hands Tracking etc.)
# accessing hand tracking wala solution (solutions.hands)
mpHands = mp.solutions.hands

# instance of hand tracker
# Hands is a class used to detect and track hands
hands = mpHands.Hands()

# reference to drawing_utils module of mediapipe
mpDraw = mp.solutions.drawing_utils # this gives us power to draw points or connect them


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    #convert BGR captured image to RGb
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #process rbg image which returns landmarks where hand is found if not found then None
    results = hands.process(imgRGB)
    
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        # two hands in one frame might be possible so draw for each hand
        for handLms in results.multi_hand_landmarks:
            # draw points (landmarks) as well as connections
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cv2.imshow('Image',img)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()




