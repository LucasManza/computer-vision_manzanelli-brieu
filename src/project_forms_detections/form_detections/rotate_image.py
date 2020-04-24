import PIL
import sys
import random

def create_rotated_image(path):
    try:
        image=PIL.Image.open(str(path))
    except IOError:
        print ('Error while opening file' + path)
        sys.exit(1)
    x=random.randint(1,360)
    new_image=image.rotate(x, PIL.Image.BICUBIC, True)
    new_image=PIL.Image.fromarray(new_image)
    new_image.save('Rot_img'+str(x)+'.jpeg')
    image.close()
    return(new_image)