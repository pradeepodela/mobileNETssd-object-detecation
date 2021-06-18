import cv2
import cvzone
import numpy as np
from numpy.core.numeric import indices
fpsReader = cvzone.FPS()
cap = cv2.VideoCapture(1)


cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)


classnames = []
thres = 0.5
nms_threshold = 0.2
classfile = r'D:\archive\Object_Detection_Files\coco.names'
configpath = r'D:\archive\Object_Detection_Files\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightspath = r'D:\archive\Object_Detection_Files\frozen_inference_graph.pb'

with open(classfile,'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')


net = cv2.dnn_DetectionModel(weightspath,configpath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)






while True:
    success, img = cap.read()
    fps, img = fpsReader.update(img,pos=(50,80),color=(0,255,0),scale=5,thickness=5)



    classid , comfs , bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(comfs).reshape(1,-1)[0])
    confs = list(map(float,comfs))


    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)


    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
        cv2.putText(img,classnames[classid[i][0]-1].upper(),(box[0]+10,box[1]+30),
        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)



        ##### NON NMS #####

    # if len(classid) != 0:
    #     for classId, confidence , box in zip(classid.flatten(),comfs.flatten(),bbox):
    #         cv2.rectangle(img,box,color=(0,255,0),thickness=2)
    #         cv2.putText(img,classnames[classId-1].upper(),(box[0]+10,box[1]+30),
    #         cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    #         cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
    #         cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)



    cv2.imshow("output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
