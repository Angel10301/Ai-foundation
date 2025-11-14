# Face Detection Module using MediaPipe
# Provides a reusable FaceDetector class for detecting faces in images/videos

import cv2          
import mediapipe as mp 
import time       

class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection  # MediaPipe face detection module
        self.mpDraw = mp.solutions.drawing_utils           # MediaPipe drawing utilities
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)  # Face detector instance

    def findFaces(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        #print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                #print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2) #Putting the fps text in the video

        return img, bboxs
    # The box around the faces and its marks. 
    def fancyDraw(self, img, bbox, l=30, t=7, rt= 1):
        x, y, w, h= bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        #Top left x,y 
        cv2.line(img, (x, y), (x + l, y),(255, 0, 255), t)
        cv2.line(img, (x, y), (x, l + y),(255, 0, 255), t)
        #Top right x1, y
        cv2.line(img, (x1, y), (x1 - l, y),(255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, l + y),(255, 0, 255), t)
        #Bottom left x,y1 
        cv2.line(img, (x, y1), (x + l, y1),(255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l),(255, 0, 255), t)
        #Bottom right x1, y1
        cv2.line(img, (x1, y1), (x1 - l, y1),(255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l),(255, 0, 255), t)
        return img 
  
def main():
    cap = cv2.VideoCapture(r'C:\Users\advan\Documents\Ai-foundation\CompVisPython\FaceDetection\FaceVideos\7.mp4')
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, Bboxs = detector.findFaces(img) # can turn off by simply putting False in () 
        print(Bboxs)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
        pTime = cTime
        
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2) #Putting the fps text in the video
        cv2.imshow("image", img)
        cv2.waitKey(1)# Control framerate of the vidoes

if __name__ == "__main__":
    main()