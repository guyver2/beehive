import numpy as np
import math
import cv2
import sys
from collections import deque

from munkres import linear_assignment



# statistical background model
class BGmodel(object):
    def __init__(self, size):
        self.size = size
        self.hist = np.zeros((self.size,480,640))
        self.model = None
        self.cpt = 0
        self.ready = False

    # add a frame and update the model
    def add(self, frame, updateModel = True):
        self.hist[self.cpt,:,:] = np.copy(frame)
        self.cpt += 1
        if self.cpt == (self.size - 1) :
            self.ready = True
        self.cpt %= self.size
        if updateModel:
            self.updateModel()
    
    def updateModel(self):
        self.model = np.median(self.hist, axis=0).astype(np.int32)
    
    def getModel(self):
        return np.copy(self.model)
       
    def apply(self, frame):
        self.add(frame)
        res = 2*(np.abs(frame-self.model).astype(np.int32))
        res = np.clip(res, 0, 255)
        return res.astype(np.uint8)


#-----------------------------------------------------------

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.filterByColor = False;
#params.minThreshold = 10;
#params.maxThreshold = 255;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 10
params.maxArea = 400
 
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.70
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.70
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1
params.maxInertiaRatio = 0.5
 
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)



class Bee(object):
    staticID = 0
    BEES_IN = 0
    BEES_OUT = 0
    def __init__(self, pos):
        self.pos = [list(pos)]
        self.lastSeen = 0
        self.ID = Bee.staticID
        Bee.staticID += 1
        self.age = 0
        self.color =  tuple(255*(0.2+4*np.random.random((3))/5))
    
    def move(self, pos):
        self.age += 1
        self.pos.append(list(pos))
    
    def pop(self):
        self.age -= 1
        self.pos.pop()
    
    def dist(self, pt, offset = 0):
        return math.sqrt((pt[0]-self.pos[-(1+offset)][0])**2 + (pt[1]-self.pos[-(1+offset)][1])**2)
        
    def draw(self, img):
        #print "drawing bee#%d"%self.ID
        cv2.polylines(img, [np.int32(self.pos)], False, self.color)



class Hive(object):
    def __init__(self, x, y, w, h):
        self.IN = 0
        self.OUT = 0
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def append(self, pt):
        # if a bee pops out from the hive entrance, then count it
        if pt[0] > self.x and pt[0] < self.x+self.w and pt[1] > self.y and pt[1] < self.y+self.h:
            self.OUT += 1
    
    def remove(self, pt):
        # if a bee desapears closer to the hive entrance, then count it
        if pt[0] > self.x and pt[0] < self.x+self.w and pt[1] > self.y and pt[1] < self.y+self.h:
            self.IN += 1
            
    def draw(self, img):
        cv2.putText(img,"IN  : %d OUT : %d"%(self.IN, self.OUT), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
        


# input video file
cap = cv2.VideoCapture('bees1.h264')
# morphological structuring element to clean the image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# background model 
bgm = BGmodel(30)
frameid = -1

bees = []
hive = Hive(53,412,582,62)
THRESHBEE = 60

try :
    while(cap.isOpened()):
        frameid += 1
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480), 0, 0, cv2.INTER_CUBIC);
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5),0).astype(np.int32)
        
        
        
        fgmask = bgm.apply(gray)
        
        #ret,fgmask = cv2.threshold(fgmask,80,255,cv2.THRESH_BINARY)
        #fgmask = cv2.erode(fgmask,kernel,iterations = 1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        
        
        im_with_keypoints = None
        
        # blob detection
        if bgm.ready :
            keypoints = detector.detect(fgmask)
            # Draw detected blobs as red circles.
            im_with_keypoints = cv2.drawKeypoints(fgmask, keypoints, np.array([]), (0,0,255))
            if len(bees) == 0:
                for kp in keypoints:
                    bees.append(Bee(kp.pt))
                    hive.append(kp.pt)
            else :
                # MUNKRES assignment, slightly better
                if True :
                    freeBees = [True for i in xrange(len(bees))]
                    freeKP = [True for i in xrange(len(keypoints))]
                    cost = np.zeros((len(keypoints), len(bees)))
                    for i,kp in enumerate(keypoints):
                        for j,b in enumerate(bees):
                            cost[i,j] = b.dist(kp.pt)
                    assignment = linear_assignment(cost)
                    for ass in assignment :
                        if cost[ass[0], ass[1]] < THRESHBEE:
                            bees[ass[1]].move(keypoints[ass[0]].pt)
                            freeBees[ass[1]] = False
                            freeKP[ass[0]] = False
                    for i in xrange(len(freeBees)): # lost bees
                        if freeBees[i]:
                            bees[i].lastSeen += 1
                    for i in xrange(len(freeKP)): # new keypoints
                        if freeKP[i]:
                            bees.append(Bee(keypoints[i].pt))
                            hive.append(kp.pt)
                    
                else :
                    #naiv assignment (kinda work)
                    newbees = []
                    freeBees = [True for i in xrange(len(bees))]
                    for kp in keypoints:
                        dists = [b.dist(kp.pt) for b in bees]
                        minpos = np.argmin(dists)
                        if dists[minpos] > THRESHBEE :
                            continue
                        if not freeBees[minpos] :
                            if dists[minpos] < bees[minpos].dist(kp.pt,1):
                                bees[minpos].pop()
                            else :
                                newbees.append(Bee(kp.pt))
                        freeBees[minpos] = False
                        bees[minpos].move(kp.pt)
                    for i in xrange(len(freeBees)):
                        if freeBees[i]:
                            bees[i].lastSeen += 1
                    bees.extend(newbees)

            # remove lost bees   
            tmp = []
            for b in bees:
                if b.lastSeen < 15: # bee still "alive"
                    tmp.append(b)
                    b.draw(im_with_keypoints)
                    b.draw(frame)
                else :
                    hive.remove(b.pos[-1])            
            bees = tmp
            hive.draw(im_with_keypoints)
            hive.draw(frame)
            print "frame : %d /  IN : %d /  OUT : %d"%(frameid, hive.IN, hive.OUT)
            cv2.imwrite("out/frame_%05d.jpg"%frameid, frame)
                    
                

       

        #cv2.imshow('frame',np.concatenate((fgmask, gray2), axis=1))
        #cv2.imshow('model',bgm.model.astype(np.uint8))
        #cv2.imshow('frame',gray.astype(np.uint8))
        #cv2.imshow('mask',fgmask)
        #if im_with_keypoints is not None : 
        #    cv2.imshow("Keypoints", im_with_keypoints)
        #cv2.imshow('color',frame)
        
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
except :
    pass
finally :
    cap.release()
    cv2.destroyAllWindows()

print "IN : ", hive.IN
print "OUT: ", hive.OUT

