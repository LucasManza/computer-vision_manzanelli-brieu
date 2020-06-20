import cv2 as cv
#import imutils

video = cv.VideoCapture("carsRt9_3.avi")
backgroundsub=cv.bgsegm.createBackgroundSubtractorMOG()
tracker=cv.TrackerKCF_create()
trackers = cv.MultiTracker_create()
trig=True
count=0

def denoise(frame, method, radius):
    kernel = cv.getStructuringElement(method, (radius, radius))
    opening = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    return closing

while True:#Repeat for each frame of the video
    frame=video.read()[1]
    
    if frame is None:#Test if we reached the end of the video
            break
        
    #frame = imutils.resize(frame, width=600)
    cv.imshow("original",frame)#Showing the video
    
    if trig == True:#Initialization process
        
        extract=backgroundsub.apply(frame)#Extracting moving objects
        nfextract=denoise(extract,cv.MORPH_ELLIPSE, 3)#Ellipse of radius=3 offer the best results for this video
        cv.imshow("movimiento",nfextract)
        
        components=cv.connectedComponentsWithStats(nfextract)#Label and info on moving objects
        #print(components)
        N=components[0]#Number of detected objects + background
        #cv.imshow("comp", cv.connectedComponents(nfextract)[1])
        
        stats=[]
        centroids=[]
        for i in range(N):
            if i!=0 and components[2][i,4]>=15:#We don't take background and objects that are too small
                stats.append(components[2][i,:])#Extract objects
                centroids.append(components[3][i,:])
                
        bBox=[]
        print(len(stats))
        for k in range(len(stats)):#Init multitracker
            (x, y, w, h) = [int(a) for a in stats[k][:4]]
            box=(x, y, x + w, y + h)
            #print(box)
            bBox.append(box)
            #cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            trackers.add(tracker, frame, box)#Weird but made to avoid an error
        trig=False#Wont do it again for 10 frames (1sec)
        
    #Grab the updated bounding box coordinates (if any) for each
    #object that is being tracked
    (success, bBox) = trackers.update(frame)
    
    for box in bBox:
            (x, y, w, h) = [int(v) for v in box]
            cv.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

    #Verify if it's time to re initialize the tracker
    count+=1
    if count>=30:
        count=0
        trig=True
    #Show the output frame
    cv.imshow("Result", frame)
            
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        print("Keyboard interrupt")
        break
video.release()
cv.destroyAllWindows()