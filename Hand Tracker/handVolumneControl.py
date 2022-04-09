import numpy as np
import cv2
import handtrackmodule as ht
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

########################
wCam, hCam = 640, 480
########################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = ht.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
length = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList)!=0:

        # Keep index 1 for x and 2 for y only if you're tracking one hand. If you are tracking more than one hand then the indices will be 2 and 3 respectively.
        x1, y1 = lmList[4][1], lmList[4][2] # x and y coordinates of thumb's tip
        x2, y2 = lmList[8][1], lmList[8][2] # x and y coordinates of index finger's tip
        cx, cy = (x1+x2) // 2, (y1+y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # distance between two coordinates = sqrt [  (x2 - x1)^2 + (y2 - y1)^2 ]
        length = math.hypot(x2-x1, y2-y1)

        # Hand Range : 50 - 300
        # Volumne Range : -65 - 0

        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    volBar = np.interp(length, [50, 300], [400, 150])
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)

    volPer = np.interp(length, [50, 300], [0, 100])

    cv2.putText(img, f'{int(volPer)} %', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

