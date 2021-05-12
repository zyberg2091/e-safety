import cv2 
import math
import time
from age_detection_mod.age_detection_utils import age_gender_detector



def age_detector(video_file):
    inputs = cv2.VideoCapture(video_file)
    age_n,loop=0,0
    alpha,age_n_list=[],[]
    limit_frame_val=200
    while True:
        ret,current_frame = inputs.read()
        loop+=1
        if loop <= limit_frame_val:
            output,age_n,boxes  = age_gender_detector(current_frame,loop)
            age_n_list.append(age_n)
            if loop==limit_frame_val and boxes>30:
                alpha=['age<18' if age == '(8-12)' or age == '(15-20)' or age == '(4-6)' or age == '(0-2)' else 'age>18' for age in age_n_list]
                val='age<18' if alpha.count('age<18')>0.8*alpha.count('age>18') else 'age>18'
                print('Frames passed : ',loop)
                return val
            elif loop==limit_frame_val and boxes<30: 
                limit_frame_val +=50


        cv2.imshow("Webcam Video",output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break