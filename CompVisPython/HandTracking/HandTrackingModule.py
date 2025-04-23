import cv2
import mediapipe as mp
import time


#Create class to put parameters
class handDetector():
    def __init__(self, mode = False, maxHands = 2, complexcity = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode #Providing the value of the mode
        self.maxHands = maxHands
        self.complexcity = complexcity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexcity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #conver to rgb
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks) #shows the information giving

        if self.results.multi_hand_landmarks: # Extract information of each hand
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNum = 0, draw = True):

        lmList = [] #Land mark list to return the postion of the marks
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNum] #Get first hand, then within it get all land marks
            for id, lm in enumerate(myhand.landmark):#Landmark what we get with the index number of our finger marks.
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (0, 0, 255), cv2.FILLED)
            
        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0) # Captures the vidoe cam live footage.
    detector = handDetector() # Create object to find hand, wont give parametters since already put in the class.

    while True:
        success, img = cap.read()
        img = detector.findHands(img)# Can put draw = Flase if i want to remove it
        lmList = detector.findPosition(img) # Can put draw = Flase if i want to remove it
        if len(lmList) != 0:
            print(lmList[4]) # This picks the land mark we want to utilize and or target
            # 0 is at wrist and goes to 1 from the bottom thumb to 4,then 5 to 8 on the next finger, then 9 - 12 on the middle finger
            # 13 - 16 on the next finger then 17-20 on the pinky finger 
            # all starting from the bottom of each finger.
        
        #Math to Displays the FPS 
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        #Display FPS
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

#If called it will run
if __name__ == "__main__":
    main()