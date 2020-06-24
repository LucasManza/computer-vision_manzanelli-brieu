import cv2 as cv
#import imutils

video = cv.VideoCapture("carsRt9_3.avi")
tracker=cv.TrackerKCF_create()
trackers = cv.MultiTracker_create()
prev=video.read()[1]


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


while True:
    frame=video.read()[1]
    
    if frame is None:#Test if we reached the end of the video
            break
    
    img_mov=movement(prev,frame)
    noisefree_img=denoise(img_mov,cv.MORPH_RECT, 2, 11)
    k=cv.getStructuringElement(cv.MORPH_RECT, (12,12))
    binary=cv.dilate(noisefree_img, k)#results amplification
    cv.imshow("binary",binary)
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours=[]
    for c in contours:#Filter smaller contours to avoid false positives
        if cv.contourArea(c)>900:
            filtered_contours.append(c)
    frame=cv.drawContours(frame, filtered_contours, -1, (0,255,0))
    cv.imshow("contours",frame)
    
    prev=frame
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:#Quit
        print("Keyboard interrupt")
        break
video.release()
cv.destroyAllWindows()