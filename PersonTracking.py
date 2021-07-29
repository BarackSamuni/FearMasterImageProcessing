import cv2
import itertools
import math

class Track:
    def __init__(self):
        self.trackerList = []
    
        self.trackerList.append(cv2.legacy.TrackerCSRT_create)
        self.trackerList.append(cv2.legacy.TrackerMedianFlow_create)
        self.trackerList.append(cv2.legacy.TrackerKCF_create)
        self.trackersObjects={}
        self.centroids = {}
        self.idCount = 0
        self.objectsBbsAndIds = {}


        

  

    def initTrackers(self,initFrame,bbox):
        x,y,w,h = bbox
        objTrackers = []

        for tracker in self.trackerList:
            trackerObj = tracker()
            trackerObj.init(initFrame,bbox)
            objTrackers.append(trackerObj)
        
        self.trackersObjects[self.idCount] = objTrackers
        self.centroids[self.idCount] = self.findCentroid(bbox)
        self.objectsBbsAndIds[self.idCount] = [x,y,w,h]
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
            
            if distance < 100:               #this is the same object,just update its centroid
                self.centroids[id] = centroid
                newObjectDetected = False
                break
        
        return newObjectDetected




    
    
    def drawBox(self,img,bbox):
        x, y, w, h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        cv2.rectangle(img,(x,y),((x+w),(y+h)),(0,255,0),3,1)
    
    def updateTrackers(self,frame):
        update = True
        IDsToRemove = []

        for idx,trackersContainer in self.trackersObjects.items():

            for tracker,flag in zip(trackersContainer,[True,False,False]):
                if flag:
                    success , bbox = tracker.update(frame)  #take only CSRT bbox
                else:
                    success , _ = tracker.update(frame)
                
                update &= success

            if not update:                              #add an id to remove
                IDsToRemove.append(idx)


            else:                                       #update object bbox and centroid
                x , y, w , h = bbox
                self.objectsBbsAndIds.update({idx: [x,y,w,h]})
                self.centroids.update({idx : self.findCentroid(bbox)})
        
        for id in IDsToRemove:                          #remove object,its trackers and centroid
            self.objectsBbsAndIds.pop(id)
            self.centroids.pop(id)
            self.trackersObjects[id].clear()
            self.trackersObjects.pop(id)



    def DisplayObjects(self,frame):
        for id,object in self.objectsBbsAndIds.items():
            x , y, w ,h = object
            self.drawBox(frame,(x,y,w,h))
            cv2.putText(frame,str(id),(int(x),int(y)-15),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    



    






            

        


    
        




        
       
            


        
        
            