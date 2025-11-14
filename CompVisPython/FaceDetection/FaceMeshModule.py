# Face Mesh Detection Module using MediaPipe
# Detects and tracks facial landmarks in real-time video

import cv2 
import mediapipe as mp
import time


class FaceMeshDetector():

    def __init__(self, static_image_mode=False, max_num_faces=2, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Store configuration parameters
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe drawing and face mesh solutions
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        
        # Create face mesh detector with specified parameters
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # Define drawing specifications for mesh visualization
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self,img,draw = True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image to detect face landmarks
        self.results = self.faceMesh.process(self.imgRGB)
        
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                                    self.drawSpec, self.drawSpec)
                
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x* iw), int(lm.y*ih)
                    # To visualy show the points(ID) on face.
                    #cv2.putText(img,str(id), (x,y),cv2.FONT_HERSHEY_PLAIN, 0.5, (0,255,0), 1)
                    #print(id,x,y)
                    face.append([x,y])
                faces.append(face)
        else:
            print("No Face Detected......")   
        
        return img, faces

def main():
    cap = cv2.VideoCapture(r'C:\Users\advan\Documents\Ai-foundation\CompVisPython\FaceDetection\FaceVideos\6.mp4')
    pTime = 0
    
    # Create face mesh detector instance
    detector = FaceMeshDetector()
    
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        
        # Print number of detected faces
        if len(faces) != 0:
            print(len(faces))
        
        # Calculate and display FPS
        cTime = time.time()
        delta = cTime - pTime
        if delta > 0:
            fps = 1 / delta
        else:
            fps = 0
        pTime = cTime
        
        # Display FPS on image
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
        
        # Show the processed image
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    

if __name__ == "__main__" :
    main()