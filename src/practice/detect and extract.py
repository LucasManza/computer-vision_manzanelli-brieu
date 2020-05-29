import cv2 as cv
import numpy as np

#Trackbar parameters
max_value = 255
max_value_H = 360//2
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
    cv.setTrackbarPos("Lowest_H_value", window_name, low_H)
    
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos("Highest_H_value", window_name, high_H)
    
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos("Lowest_S_value", window_name, low_S)
    
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos("Highest_S_value", window_name, high_S)
    
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos("Lowest_V_value", window_name, low_V)
    
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos("Highest_V_value", window_name, high_V)
    
cv.namedWindow(window_name)

cv.createTrackbar("Lowest_H_value", window_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar("Highest_H_value", window_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar("Lowest_S_value", window_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar("Highest_S_value", window_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar("Lowest_V_value", window_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar("Highest_V_value", window_name , high_V, max_value, on_high_V_thresh_trackbar)

def detect(image):
    frame_HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    return(frame_threshold)

def extract(image):
    frame_gray=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret,labels,stats,centroids=cv.connectedComponentsWithStats(frame_gray)
    areas=stats[:,4]
    k=np.where(areas==max(areas))#Find connected comp with max area
    k=zip(k)
    x=centroids[k,0]
    y=centroids[k,0]
    center=(x,y)#Extract it's center
    area=areas[k]#Extract it's area
    return(center,area)