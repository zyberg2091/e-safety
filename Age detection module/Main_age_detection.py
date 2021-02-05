# Import required modules

import cv2 
import math
import time

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (80.4263377603, 97.7689143744, 115.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

padding = 20
loop,age=0,0
boxes=0


def age_gender_detector(frame,loop):
    # Read frame
    global boxes
    global age
    frameFace, bboxes = getFaceBox(faceNet, frame)
    for bbox in bboxes:
        boxes+=1

        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
          
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        label = "{},{}".format(gender, age)
        cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return frameFace,age,boxes

inputs = cv2.VideoCapture('test_vids/vid7.mp4')
age_n=0
alpha=[]
age_n_list=[]
limit_frame_val=200
while True:
    ret,current_frame = inputs.read()
    loop+=1
    if loop <= limit_val:
        output,age_n,boxes  = age_gender_detector(current_frame,loop)
        age_n_list.append(age_n)
        if loop==limit_val and boxes>30:
            alpha=['age<18' if age == '(8-12)' or age == '(15-20)' or age == '(4-6)' or age == '(0-2)' else 'age>18' for age in age_n_list]
            print('age<18' if alpha.count('age<18')>0.8*alpha.count('age>18') else 'age>18')
            print('Frames passed : ',loop)
            break
        elif loop==limit_val and boxes<30: 
            limit_val +=50

    cv2.imshow("Webcam Video",output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
