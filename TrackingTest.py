from PersonTracking import Obsticle, Track
import cv2
import json
import time
import csv

VIDEOILE = "video1.mp4"
JSONFILE = "detections.json"
CSVFILE = "coordinates2.csv"

with open(JSONFILE) as json_file:
    coordinates= json.load(json_file)

vid = cv2.VideoCapture(VIDEOILE)
fps= int(vid.get(cv2.CAP_PROP_FPS))

outputvid = cv2.VideoWriter("tracking.avi",cv2.VideoWriter_fourcc(*'DIVX'),vid.get(cv2.CAP_PROP_FPS),(416,416))
success = True
allTrackers = Track()
framecnt = 0

with open(CSVFILE,'w',newline='') as f:
    writeCSV = csv.writer(f)
    header = ['ID','Time[s]','Centroid']
    writeCSV.writerow(header)

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

        timeStamp = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        factor = int(0.25* vid.get(cv2.CAP_PROP_FPS))
        timeStampCheck = factor/vid.get(cv2.CAP_PROP_FPS)

        if True:
            for id , centroid in allTrackers.centroids.items():
                    data = [id , timeStamp, centroid]
                    writeCSV.writerow(data)



        if cv2.waitKey(1) & 0xff == ord('q'):
            break

vid.release()
outputvid.release()
cv2.destroyAllWindows()





