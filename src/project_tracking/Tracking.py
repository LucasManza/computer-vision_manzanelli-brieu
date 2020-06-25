import cv2 as cv
import numpy as np
#import imutils

video = cv.VideoCapture("carsRt9_3.avi")
prev=video.read()[1]
tracked=None
count=1


def movement(image1,image2):#Gives back a binary image of what moved between two frames
    gray1=cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray2=cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    
    diff=cv.absdiff(gray1, gray2)
    _,tresh=cv.threshold(diff, 60, 255, cv.THRESH_BINARY)
    return(tresh)

def denoise(frame, method, radius1, radius2):#Clear noise on an image
    kernel1 = cv.getStructuringElement(method, (radius1, radius1))
    kernel2 = cv.getStructuringElement(method, (radius2, radius2))
    opening = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel1)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel2)
    return closing

def get_distance(bbox1, bbox2):#Computes distance in pixels between two bounding boxes
    (x1, y1, w1, h1) = bbox1
    (x2, y2, w2, h2) = bbox2
    
    cx1=int(x1 + (w1/2))
    cx2=int(x2 + (w2/2))
    cy1=int(y1 + (h1/2))
    cy2=int(y2 + (h2/2))
    
    distance=np.sqrt((cx2-cx1)**2+(cy2-cy1)**2)
    return(abs(distance))

def add_or_update(contours,tracked, image):
    new_index=len(tracked)+1
    for t in tracked:
        tracker=t["tracker"]
        success,t["bbox"]=tracker.update(image)
        
    for c in contours:
        for t in tracked:#Compare each detected contour with already tracked objects
            if get_distance(c.get("bbox"),t.get("bbox"))<200:
                t["updated"]=True
                break
                    
        tracker=cv.TrackerKCF_create()#Create a new tracker
        c["tracker"]=tracker
        sucess=tracker.init(image,c["bbox"])
        if not(sucess):#Verify init
            c["updated"]=False
        c["index"]=new_index
        tracked.append(c)
    
    i=0    
    for x in tracked:
        if not(x["updated"]):#Everything that failed is removed
            tracked.remove(x)
            i+=1
    print("successfully removed")
    print(i)
    for i in range(len(tracked)):#Reorder index after removing elements
        tracked[i]["index"]=i
    return(tracked)


while True:
    frame=video.read()[1]
    print("Loop n°")
    print(count)
    
    if frame is None:#Test if we reached the end of the video
            break
    
    img_mov=movement(prev,frame)
    noisefree_img=denoise(img_mov,cv.MORPH_RECT, 2, 11)
    
    k=cv.getStructuringElement(cv.MORPH_RECT, (12,12))
    binary=cv.dilate(noisefree_img, k)#results amplification
    cv.imshow("binary",binary)
    
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours=[]
    index=1
    for c in contours:#Filter smaller contours to avoid false positives
        if cv.contourArea(c)>900:
            a={"index": index, "bbox": cv.boundingRect(c), "tracker": None, "updated": True}
            filtered_contours.append(a)
            index+=1
    print("Objects detected")
    print(len(filtered_contours))
            
    #frame=cv.drawContours(frame, filtered_contours, -1, (0,255,0))
    if tracked!=None:#Main case
        tracked=add_or_update(filtered_contours, tracked, frame)
    else:#First loop
        tracked=filtered_contours
        for t in tracked:
            tracker=cv.TrackerKCF_create()#Create a new tracker
            t["tracker"]=tracker
            sucess=tracker.init(frame,t["bbox"])
            if not(sucess):#Verify init
                t["updated"]=False
    print("Tracked objects")
    print(len(tracked))
        
    #Missing a function to draw results on frame
    cv.imshow("contours",frame)
    
    for t in tracked:
        t["updated"]=False
    prev=frame
    count+=1
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:#Quit
        print("Keyboard interrupt")
        break
video.release()
cv.destroyAllWindows()