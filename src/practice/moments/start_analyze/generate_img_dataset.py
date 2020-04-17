import imutils
import cv2
import random


def __cut_roi_img__(frame, size: int):
    height = frame.shape[0]
    width = frame.shape[1]
    min_height: int = int(height / 2 - size)
    max_height: int = int(height / 2 + size)
    min_width: int = int(width / 2 - size)
    max_width: int = int(width / 2 + size)
    roi = frame[min_height:max_height, min_width:max_width]
    cv2.imshow('', roi)
    return roi


def __resize_image__(image, fx, fy):
    return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def __rotate_image__(image, angle):
    return imutils.rotate(image, angle)


def generate_dataset(image, img_target: str):
    for i in range(1, 10):
        aux_image = __rotate_image__(image, random.randint(0, 360))
        # aux_image = __resize_image__(image, 100, 50)
        src_naming = img_target + str(i) + '.png'
        cv2.imwrite(src_naming, aux_image)


if __name__ == '__main__':
    star_img = cv2.imread('../../../assets/star-image.png')
    cv2.imshow('', star_img)
# target: str = '../../../assets/dataset/start/'
# generate_dataset(star_img, target)
