import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) # Captures the vidoe cam live footage.

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #conver to rgb
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks) #shows the information giving

    if results.multi_hand_landmarks: # Extract information of each hand
        for handLms in results.multi_hand_landmarks:
           for id, lm in enumerate(handLms.landmark):#Landmark we get with the index number of our finger marks.
               # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 4: # Determines where and how many big pink dots there are.
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    #Math to Displays the FPS 
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #Display FPS
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)