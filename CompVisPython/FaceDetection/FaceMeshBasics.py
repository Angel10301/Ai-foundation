import cv2 
import mediapipe as mp
import time

cap = cv2.VideoCapture(r'C:\Users\advan\Documents\Ai-foundation\CompVisPython\FaceDetection\FaceVideos\6.mp4')
pTime = 0

while True:
    success, img = cap.read()
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime  
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2) #Putting the fps text in the video
    cv2.imshow("Image", img)
    cv2.waitKey(1)