import cv2
import json
from tracker import *
import time

def drawBox(img,bbox):
    x, y, w, h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(0,255,0),3,1)

with open('detections.json') as json_file:
    bboxes= json.load(json_file)

vid = cv2.VideoCapture("OriginalCleanVideo.MOV")
fps= int(vid.get(cv2.CAP_PROP_FPS))


# for tracking
bbox = bboxes["detections"][0]["objects"][0]
success , frame = vid. read()
height,width,_ = frame.shape
size = (width,height)
bbox = (int(bbox["left"]),int(bbox["top"]),int(bbox["right"])-int(bbox["left"]),int(bbox["bottom"])-int(bbox["top"]))
tracker1 = cv2.TrackerCSRT_create()
tracker2 = cv2.legacy.TrackerMedianFlow_create()
tracker3 = cv2.legacy.TrackerKCF_create()

#bbox =cv2.selectROI("frame",frame,False,False)
frame=cv2.resize(frame,(416,416))
cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,255,0),3,1)
#frame=cv2.resize(frame,size)
cv2.imshow("frame",frame)
cv2.waitKey()
tracker1.init(frame,bbox)
tracker2.init(frame,bbox)
tracker3.init(frame,bbox)

while success:
    success , frame = vid. read()

    if success:
        frame=cv2.resize(frame,(416,416))
        #time.sleep(1/fps)
       

        # track the object
        updated1 , bbox = tracker1.update(frame)
        updated2,_ = tracker2.update(frame)
        updated3,_ = tracker3.update(frame)

        if updated1 and updated2 and updated3:
            drawBox(frame,bbox)
        else:
            bbox = None
      
    else:
        break

    cv2.imshow("video",frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()