import cv2


def hu_moments(binary_img):
    moments = cv2.moments(binary_img)
    return cv2.HuMoments(moments)