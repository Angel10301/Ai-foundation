import cv2
import mediapipe as mp
import time 
# Open video file using OpenCV's Video Capture
cap = cv2.VideoCapture(r'C:\Users\advan\Documents\Ai-foundation\CompVisPython\FaceDetection\FaceVideos\1.mp4')
pTime = 0

# Initialzing mediapipe solutions for face detection and drawing utilities
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

# Main loop to process each frame
while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converts image from BGR to RGB
    results = faceDetection.process(imgRGB)
    print(results)

    # Check if any faces were detected in the frame
    if results.detections: 
        for id, detection in enumerate(results.detections): # Iterate over eac detected face
            mpDraw.draw_detection(img, detection)
            #print(id, detection)
            #print(detection.score)
            #print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2) #Putting the fps text in the video


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2) #Putting the fps text in the video
    cv2.imshow("image", img)
    cv2.waitKey(1)# Control framerate of the vidoe
