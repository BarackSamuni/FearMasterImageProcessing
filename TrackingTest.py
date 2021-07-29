from PersonTracking import Track
import cv2
import json
import time

with open('detections.json') as json_file:
    coordinates= json.load(json_file)

vid = cv2.VideoCapture("OriginalCleanVideo.mp4")
fps= int(vid.get(cv2.CAP_PROP_FPS))

outputvid = cv2.VideoWriter("tracking.avi",cv2.VideoWriter_fourcc(*'DIVX'),vid.get(cv2.CAP_PROP_FPS),(416,416))
success = True
allTrackers = Track()
framecnt = 0

while success:
    success , frame = vid. read()

    if success:
        frame=cv2.resize(frame,(416,416))
        #time.sleep(1/fps)

        for info in coordinates["detections"][framecnt]["objects"]:
            bbox = (int(info["left"]),int(info["top"]),int(info["right"])-int(info["left"]),int(info["bottom"])-int(info["top"]))

            new = allTrackers.newObjectDetected(bbox)

            if new:
                allTrackers.initTrackers(frame,bbox)

            else:
                allTrackers.updateTrackers(frame)
            
        allTrackers.DisplayObjects(frame)
        
 
        framecnt += 1

        if framecnt > len(coordinates["detections"])-1:
            break



    else:
        break


    

    cv2.imshow("video",frame)
    outputvid.write(frame)



    if cv2.waitKey(1) & 0xff == ord('q'):
        break

vid.release()
outputvid.release()
cv2.destroyAllWindows()





