import cv2
import itertools
import math

Obsticle = 1

class Track:
    def __init__(self):
        self.trackerList = []
    
        self.trackerList.append(cv2.legacy.TrackerCSRT_create)
        #self.trackerList.append(cv2.legacy.TrackerMedianFlow_create)
        #self.trackerList.append(cv2.legacy.TrackerKCF_create)
        #self.trackerList.append(cv2.legacy.TrackerMOSSE_create)
        self.trackersObjects={}
        self.centroids = {}
        self.idCount = 0
        self.objectsBbsAndIds = {}



    def initTrackers(self,initFrame,bbox):
        x,y,w,h = bbox
        objTrackers = []
        area = w*h
        if area < 2500:
            return
        for tracker in self.trackerList:
            trackerObj = tracker()
            trackerObj.init(initFrame,bbox)
            objTrackers.append(trackerObj)
        
        self.trackersObjects[self.idCount] = objTrackers
        self.centroids[self.idCount] = self.findCentroid(bbox)
        self.objectsBbsAndIds[self.idCount] = [x,y,w,h,0,0]
        self.idCount += 1
        


    
    def findCentroid(self,bbox):
        x,y,w,h = bbox
        cx = (x + (x + w)) // 2
        cy = (y + (y + h)) // 2

        return (cx,cy)

    def newObjectDetected(self,bbox):
        centroid = self.findCentroid(bbox)
        newObjectDetected = True

        for id, center in self.centroids.items():
            distance = math.hypot(centroid[0]-center[0],centroid[1]-center[1])
            
            if distance < 40:               #this is the same object,just update its centroid
                self.centroids[id] = centroid
                newObjectDetected = False
                break
        
        return newObjectDetected 





    
    
    def drawBox(self,img,bbox):
        x, y, w, h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        cv2.rectangle(img,(x,y),((x+w),(y+h)),(0,255,0),3,1)
        tmp = self.findCentroid(bbox)
        center = (int(tmp[0]),int(tmp[1]))
        cv2.circle(img,center,5,color=(0, 0, 255),thickness=5)
    
    def updateTrackers(self,frame):
        update = True
        IDsToRemove = []
        Static = False
        global Obsticle
        for idx,trackersContainer in self.trackersObjects.items():

            for tracker,flag in zip(trackersContainer,[True,False,False]):
                if flag:
                    success , bbox = tracker.update(frame)  #take only CSRT bbox
                else:
                    success , _ = tracker.update(frame)
                
                update &= success
            
            centroidBbox = self.findCentroid(bbox)
            distance = math.hypot(self.centroids[idx][0] - centroidBbox[0] ,self.centroids[idx][1] - centroidBbox[1] )

            if distance < 5:
                self.objectsBbsAndIds[idx][5] += 1
                Static = True

            if idx == 44 or idx == 71:
                Obsticle = idx
            

            if self.objectsBbsAndIds[idx][5]==120 and idx !=  Obsticle:
                IDsToRemove.append(idx)
            
            if not update:                              #add an id to remove
                self.objectsBbsAndIds[idx][4] += 1
                if self.objectsBbsAndIds[idx][4] == 80 and self.objectsBbsAndIds[idx][5]<120:
                    IDsToRemove.append(idx)



            else:                                       #update object bbox and centroid
                x , y, w , h = bbox
                self.objectsBbsAndIds.update({idx: [x,y,w,h,0,self.objectsBbsAndIds[idx][5]*Static]})
                self.centroids.update({idx : self.findCentroid(bbox)})
        
        for id in IDsToRemove:                          #remove object,its trackers and centroid
            self.objectsBbsAndIds.pop(id)
            self.centroids.pop(id)
            self.trackersObjects[id].clear()
            self.trackersObjects.pop(id)


    def DisplayObjects(self,frame):
        for id,object in self.objectsBbsAndIds.items():
            x , y, w ,h , _ , _  = object
            self.drawBox(frame,(x,y,w,h))
            cv2.putText(frame,str(id),(int(x),int(y)-15),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)