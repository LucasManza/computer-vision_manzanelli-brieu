import cv2 as cv
import numpy as np

realLength=0.3#Change this according to the reference placed in the image (in meters)
points = []

def order_points_2(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-most point,
    # the second entry is the bottom-most point
    rect = np.zeros((2, 2), dtype="float32")
    # the top point will have the smallest y, whereas
    # the bottom point will have the largest y
    Y=pts[:,1]
    index_top=np.where(Y==min(Y))
    index_bot=np.where(Y==max(Y))
    rect[0] = tuple(pts[int(index_top[0])])
    rect[1] = tuple(pts[int(index_bot[0])])
    # return the ordered coordinates
    return rect

def on_click(event, x, y, flag, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        global points
        points.append((x, y))

def get_system_and_scale(image, image_name):
    cv.setMouseCallback(image_name, on_click)
    global points
    if cv.waitKey() & 0xff==13:#Press enter to validate
        if len(points) == 2:
            pts = np.array(points, dtype="float32")
            rect = order_points_2(pts)
            #Draw coordinate system on image
            image=cv.arrowedLine(image, tuple(rect[0]), tuple(rect[1]), (0,0,255))#Draw first vector alongside the reference
            a=rect[1,1]-rect[0,1]
            b=rect[0,0]-rect[1,0]
            n=(a,b)#Compute normal vector
            image=cv.arrowedLine(image, tuple(rect[0]), (rect[0][0]+n[0],rect[0][1]+n[1]), (0,0,255))#Draw normal vector
            origin=tuple(rect[0])
            vect=(rect[1][0]-rect[0][0],rect[1][1]-rect[0][1])
            norm=np.sqrt(vect[0]**2+vect[1]**2)
            scale_factor=norm/realLength#Pixel/meters
            if cv.waitKey() & ord('r'):#Work in Progress
                image=cv.rectangle(image, origin, (rect[0,0]-b+a,rect[0,1]+a+b), (255,0,0), 10)#Draw patern
            return(image,origin, scale_factor)
    else:
         return("error, you need to select 2 pts")
                
    
#Test routine:
frame=cv.imread("Test.jpg")
cv.imshow("frame",frame)
result=get_system_and_scale(frame, "frame")
if isinstance(result,tuple):
    cv.imshow("frame",result[0])
    print(result[1],result[2])
else:
    print(result)
cv.waitKey()
cv.destroyAllWindows()