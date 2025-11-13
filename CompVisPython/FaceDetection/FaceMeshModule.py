import cv2 
import mediapipe as mp
import time


class FaceMeshDectector():

    def __init__(self, staticMode = False, maxFaces = 2, minDetectionCon =0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon)
        self. drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius = 1)

    def findFaceMesh(self,img,draw = True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                                    self.drawSpec, self.drawSpec)
                #for id, lm in enumerate(faceLms.landmark):
                    #print(lm)
                   # ih, iw, ic = img.shape
                    #x, y = int(lm.x* iw), int(lm.y*ih)
                    #print(id,x,y)
                 
        #else:
            #print("No Face Detected......")   
        return img

def main():
    cap = cv2.VideoCapture(r'C:\Users\advan\Documents\Ai-foundation\CompVisPython\FaceDetection\FaceVideos\4.mp4')
    pTime = 0
    detector = FaceMeshDectector()
    while True:
        success, img = cap.read()
        img = detector.findFaceMesh(img)
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
    

if __name__ == "__main__" :
    main()