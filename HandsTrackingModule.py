import mediapipe as mp
import cv2

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode=mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        #self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils 

    def findHands(self, img, draw=True):
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    

    def findPositions(self, img, handNo=0, draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks:
            selectedHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(selectedHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw and id==8:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

        return lmList

def main(): 
    cap = cv2.VideoCapture(0)  
    obj = handDetector()
    
    while True:
        success, img = cap.read()
        img=obj.findHands(img)

        lmList = obj.findPositions(img)
        print(lmList)

        cv2.imshow('Image',img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
