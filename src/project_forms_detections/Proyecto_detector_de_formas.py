#import section
import cv2 as cv
import numpy as np

#Variables section
blockSizeMaxGauss=150

max_elem = 2#Max number of diferent types of elements for morphologic image_operators
max_kernel_size = 50

title_trackbar_kernel_size1 = 'Opening Kernel size:\n 2n +1'
title_trackbar_kernel_size2 = 'Closing Kernel size:\n 2n +1'
title_opening_window = 'Opening'
title_closing_window = 'Closing'
title_trackbar_kernel_type = 'Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'

#Webcam init
cap=cv.VideoCapture(0)
ret, frame=cap.read()
image = frame

#Primer paso: Convertir la imagen a monocromáticaq
def convGray(img):
    return(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    
#Segundo paso: Aplicar un threshold con umbral ajustable con una barra de desplazamiento
def Gauss(val,img=image):
    gray = convGray(img)
    th2 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,val,0)
    cv.imshow('Gauss', th2)
    return(th2)
    
cv.namedWindow('Gauss')
cv.createTrackbar('Trackbar', 'Gauss', 3, blockSizeMaxGauss, Gauss)

Gauss(3,image)
cv.waitKey()
P2=Gauss(cv.getTrackbarPos('Trackbar', 'Gauss'),image)
    
#Tercero paso: Aplicar operaciones morfológicas para eliminar ruido de la imagen
def opening(val,img=P2):
    kernel_size1 = cv.getTrackbarPos(title_trackbar_kernel_size1, title_opening_window)
    kernel_type1 = 0
    
    #Opening type selection
    val_type1 = cv.getTrackbarPos(title_trackbar_kernel_type, title_opening_window)
    if val_type1 == 0:
        kernel_type1 = cv.MORPH_RECT
    elif val_type1 == 1:
        kernel_type1 = cv.MORPH_CROSS
    elif val_type1 == 2:
        kernel_type1 = cv.MORPH_ELLIPSE
    
    kernel1=cv.getStructuringElement(kernel_type1, (2*kernel_size1 + 1, 2*kernel_size1+1), (kernel_size1, kernel_size1))#Opening Element
    result=cv.morphologyEx(img, cv.MORPH_OPEN, kernel1)#Opening
    cv.imshow(title_opening_window, result)
    return(result)

cv.namedWindow(title_opening_window)
cv.createTrackbar(title_trackbar_kernel_type, title_opening_window , 0, max_elem, opening)
cv.createTrackbar(title_trackbar_kernel_size1, title_opening_window , 0, max_kernel_size, opening)

opening(0,P2)
cv.waitKey()
openingP3=opening(0,P2)

def closing(val,img=openingP3):
    kernel_size2 = cv.getTrackbarPos(title_trackbar_kernel_size2, title_closing_window)
    kernel_type2 = 0
        
    #Closing type selection
    val_type2 = cv.getTrackbarPos(title_trackbar_kernel_type, title_closing_window)
    if val_type2 == 0:
        kernel_type2 = cv.MORPH_RECT
    elif val_type2 == 1:
        kernel_type2 = cv.MORPH_CROSS
    elif val_type2 == 2:
        kernel_type2 = cv.MORPH_ELLIPSE
    
    kernel2=cv.getStructuringElement(kernel_type2, (2*kernel_size2 + 1, 2*kernel_size2+1), (kernel_size2, kernel_size2))#Closing Element
    result=cv.morphologyEx(img, cv.MORPH_CLOSE, kernel2)#Closing
    cv.imshow(title_closing_window, result)
    return(result)

cv.namedWindow(title_closing_window)
cv.createTrackbar(title_trackbar_kernel_type, title_closing_window , 0, max_elem, closing)
cv.createTrackbar(title_trackbar_kernel_size2, title_closing_window , 0, max_kernel_size, closing)

closing(0,openingP3)


#Quit
cv.waitKey()
cap.release()
cv.destroyAllWindows()