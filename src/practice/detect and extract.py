import cv2 as cv
import numpy as np

#Trackbar parameters
max_value = 255
max_value_H = 360
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value

window_name="Boundaries"

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos("Lowest_H", window_name, low_H)
    
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos("Highest_H", window_name, high_H)
    
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos("Lowest_S", window_name, low_S)
    
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos("Highest_S", window_name, high_S)
    
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos("Lowest_V", window_name, low_V)
    
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos("Highest_V", window_name, high_V)
    
cv.namedWindow(window_name)

cv.createTrackbar("Lowest_H", window_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar("Highest_H", window_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar("Lowest_S", window_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar("Highest_S", window_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar("Lowest_V", window_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar("Highest_V", window_name , high_V, max_value, on_high_V_thresh_trackbar)

def detect(image):
    frame_HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    return(frame_threshold)

def extract(image):
    frame=detect(image)
    [ret,labels,stats,centroids]=cv.connectedComponentsWithStats(frame)
    print(stats)
    areas=stats[:,4]
    for i in range(len(areas)):#To avoid extracting the whole image's center
        x=areas[i]
        if x>800000:
            areas[i]=0
    k=np.where(areas==max(areas))#Find connected comp with max area
    print(k)
    k=int(k[0])
    print(centroids)
    x=centroids[k,0]
    y=centroids[k,1]
    center=(x,y)#Extract it's center
    area=areas[k]#Extract it's area
    return(center,area)

#Test routine
frame=cv.imread("Test.jpg")
cv.imshow("frame", frame)
while True:
    cv.imshow("cal", detect(frame))
    if cv.waitKey(0) & 0xFF==13:
        k=extract(frame)
        print(k)
        cv.circle(frame,(int(k[0][0]),int(k[0][1])),10,(0,0,255))
        cv.imshow("circle",frame)
        cv.waitKey()
        break
cv.destroyAllWindows()
