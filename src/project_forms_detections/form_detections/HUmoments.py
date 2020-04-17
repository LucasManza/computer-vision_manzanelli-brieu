import cv2 as cv
import matplotlib.image as mpimg
#Capaz podemos usar la biblioteca xldr para pasar los momentos de Hu en un archivo Excel, lo haré después

def img_to_HU(path):
    #path test:(just in case)
    if type(path) != str:
        return("Please use a valid path name (string type)")

    image=mpimg.imread(path)#reading image
    gray_image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)#convertion to grayscale image
    _,thresh_image=cv.threshold(gray_image,127,255,cv.THRESH_BINARY)#convertion to binary image
    moments=cv.moments(thresh_image)#compute moments
    HU_moments=cv.HuMoments(moments)#compute HU moments

    #Saving in a new file, in the same directory as HUmoments.py
    with open("data.txt", "x") as file:
        for m in HU_moments:
            file.write(str(m)+"\t")#Prints each one of the HU moments of the source image, marking separations with tab

    return(HU_moments)#Just in case