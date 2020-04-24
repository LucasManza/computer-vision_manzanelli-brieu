import PIL
import sys
import random

def create_resized_image(path):
    try:
        image=PIL.Image.open(str(path))
    except IOError:
        print ('Error while opening file' + path)
        sys.exit(1)
    w,h=image.size
    x=random.randint(1,10)
    w=w*x
    h=h*x
    new_image=image.resize((w, h), PIL.Image.ANTIALIAS)
    #new_image2=PIL.Image.fromarray(new_image)
    new_image.save('Scale_img'+str(x)+'.jpeg')
    image.close()
    return(new_image)