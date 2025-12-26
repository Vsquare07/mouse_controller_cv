import cv2
from math import sqrt
import handtrackermodule
import time

from Quartz import CGEventCreateMouseEvent, CGEventPost
from Quartz import kCGEventMouseMoved, kCGEventLeftMouseDown, kCGEventLeftMouseUp
from Quartz import kCGMouseButtonLeft, kCGHIDEventTap

def move_mouse(x, y):
    event = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, event)

def left_mouse_click(x, y):
    down_event = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x,y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, down_event)

def left_mouse_lift(x, y):
    up_event = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x,y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, up_event)


pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = handtrackermodule.handDetector(maxHands=1)
while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    if not success:
        break
    img = detector.findHands(img)
    lmList = detector.findPosition(img, tracking_id=8)
    if len(lmList) != 0:
        index = lmList[8]
        thumb = lmList[4]
        pinky = lmList[20]
        move_mouse(index[1], index[2])

        distance = sqrt((pinky[1] - thumb[1])**2+(pinky[2] - thumb[2])**2)

        if(distance <= 100):
            cv2.putText(img, "Click!", (10,100) ,cv2.FONT_HERSHEY_COMPLEX, 3, (255,255,255), 3)
            left_mouse_click(index[1], index[2])
        else:
            left_mouse_lift(index[1], index[2])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, "FPS = "+str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()