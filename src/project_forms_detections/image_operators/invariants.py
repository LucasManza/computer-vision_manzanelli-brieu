# from math import copysign, log10

import cv2


def hu_invariants(contours):
    moments = cv2.moments(contours)
    return cv2.HuMoments(moments)


# def log_transform(hu_moments):
#     for i in range(0, 7):
#         hu_moments[i] = -1*