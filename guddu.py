from sklearn.neural_network import MLPClassifier
import cv2
import numpy as np
import dlib
import pyautogui
from math import *
import os
import signal,time
from sklearn.cluster import KMeans
import pafy
import vlc

url = "https://www.youtube.com/watch?v=a29jzOHwwq0"
video = pafy.new(url)
best = video.getbest()
playurl = best.url
Instance = vlc.Instance()
player = Instance.media_player_new()
Media = Instance.media_new(playurl)
Media.get_mrl()
player.set_media(Media)
def runnert():
    player.play()    
def midpoint(p1 ,p2):
    return int((p1.x - p2.x)*(p1.x - p2.x)+(p1.y + p2.y)*(p1.y + p2.y))


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\\Users\\user1\\Desktop\\hitum\\facial-landmarks\\shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier(r'D:\\python3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'D:\\python3\\Lib\\site-packages\\cv2\\data\\haarcascade_righteye_2splits.xml')

kmeans = KMeans(n_clusters=3)

di={0:"Happy",1:"Sad",2:"Angry",3:"Neutral"}

cap = cv2.VideoCapture(0)
X,y=[],[]
counter=0
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2
while True:
    print(counter)
    _, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        left_point = midpoint(landmarks.part(36), landmarks.part(36))
        right_point = midpoint(landmarks.part(39), landmarks.part(39))
        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))
        fl=midpoint(landmarks.part(21), landmarks.part(22))
        lu=midpoint(landmarks.part(50), landmarks.part(52))
        ld=midpoint(landmarks.part(56), landmarks.part(58))
        lm=midpoint(landmarks.part(48), landmarks.part(64))
        #X.append([left_point,right_point,center_top,center_bottom,fl,lu,ld,lm])
        if(counter==60):
            kmeans.fit(X)
        elif(counter>60):
            cv2.putText(frame, di[np.amax(kmeans.predict([[left_point,right_point,center_top,center_bottom,fl,lu,ld,lm]])[0])], org, font,fontScale, color, thickness, cv2.LINE_AA)
            if(np.amax(kmeans.predict([[left_point,right_point,center_top,center_bottom,fl,lu,ld,lm]])[0])==2):
                print("iwudbbbbbbbbbbbbbb")
                runnert()
                time.sleep(20)
            else:
                player.stop()
        else:
            cv2.rectangle(frame,(left_point,right_point),(center_top,center_top),(0,255,0),2)
            X.append([left_point,right_point,center_top,center_bottom,fl,lu,ld,lm])
        counter=counter+1
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
