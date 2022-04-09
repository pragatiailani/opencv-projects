import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon,
                                        self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []

        ###################################################################################################
        # THIS CODE IS FOR TRACKING JUST ONE HAND
        ###################################################################################################
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        ###################################################################################################


        ###################################################################################################
        # THIS CODE IS FOR TRACKING MORE THAN ONE HAND (I.E. TWO HANDS AS THE MAX-NUMBER OF HANDS IS TWO).
        ###################################################################################################
        # if self.results.multi_hand_landmarks:

            # myHand = []
            # for i in range(0, len(self.results.multi_hand_landmarks)):
            #     myHand.append(self.results.multi_hand_landmarks[i])

            #     for id, lm in enumerate(myHand[i].landmark):
            #         h, w, c = img.shape
            #         cx, cy = int(lm.x * w), int(lm.y * h)

            #         lmList.append([i, id, cx, cy])

            #         if draw:
            #             cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        ###################################################################################################

        return lmList


def main():

    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 35), cv2.FONT_ITALIC, 1, (235, 106, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
