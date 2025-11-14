import cv2 
import mediapipe as mp
import time

cap = cv2.VideoCapture(r'C:\Users\advan\Documents\Ai-foundation\CompVisPython\FaceDetection\FaceVideos\4.mp4')
pTime = 0

# Initialize MediaPipe face mesh components
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=7)  # Detect up to 7 faces
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)  # Drawing style for mesh


# Main processing loop
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect face landmarks
    results = faceMesh.process(imgRGB)
    
    # Process detected faces
    if results.multi_face_landmarks:
        print(f"Detected {len(results.multi_face_landmarks)} face(s)")
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,
                                  drawSpec, drawSpec)
            
            # Extract and print landmark coordinates
            for id, lm in enumerate(faceLms.landmark):
                #print(lm)
                ih, iw, ic = img.shape
                x, y = int(lm.x* iw), int(lm.y*ih)
                print(id,x,y)
                
        else:
            print("No Face Detected......")   



    cTime = time.time()
    delta = cTime - pTime
    if delta > 0:
        fps = 1 / delta
    else:
        fps = 0
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}', (20,70),cv2.FONT_HERSHEY_PLAIN,
                3, (0,255,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
