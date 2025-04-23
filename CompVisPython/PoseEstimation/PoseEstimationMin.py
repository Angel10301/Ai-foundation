import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()



cap = cv2.VideoCapture(r'')
if not cap.isOpened():# if check to see if vidoes are running properly
    print("Error: Could not open video file")
    exit()

pTime = time.time()
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 100), cv2.FILLED)


    cTime = time.time()
    fps = 1/(cTime - pTime) if (cTime - pTime) > 0 else 0 

    pTime = cTime
    #Display FPS
    # img is what is on screen, str(int) is used to display text, at the orgin, picking font from cv2, font size, color of it, 
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3) 

    cv2.imshow("Image", img)
    cv2.waitKey(1) # Can slow it down by raising the value
