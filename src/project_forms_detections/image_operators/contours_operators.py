import cv2
from enum import Enum


class DetectionContourEnum(Enum):
    EXTERNAL_CONT_DETECT = 1
    TREE_CONT_DETECT = 2


def find_contours(binary_img, detection_type: DetectionContourEnum = DetectionContourEnum.TREE_CONT_DETECT):
    if detection_type == DetectionContourEnum.EXTERNAL_CONT_DETECT:
        detection_type = cv2.RETR_EXTERNAL
    elif detection_type == DetectionContourEnum.TREE_CONT_DETECT:
        detection_type = cv2.RETR_TREE

    contours, _ = cv2.findContours(binary_img, detection_type, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours(img, contours, colour):
    copy_img = img.copy()
    cv2.drawContours(copy_img, contours, -1, colour, 2)
    return copy_img


# Apply a rectangle detection by contours, instead of drawing the contour per se.
def draw_contours_rect(img, contours, colour):
    copy_img = img.copy()
    for cont in contours:
        draw_contour_rect(copy_img, cont, colour)
    return copy_img


def draw_contour_rect(img, contour, colour):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)


def compare_contours(contour, target_contour):
    return cv2.matchShapes(contour, target_contour, cv2.CONTOURS_MATCH_I3, 0)


def filter_by_distance(contours, target_contour, match_error: float):
    """"
    Filter a list of contours by comparing distance with the target's contours.

    """
    if target_contour is None or target_contour.__len__() == 0: return []

    return list(filter(lambda x: compare_contours(x, target_contour) <= match_error, contours))


def filter_by_area(contours, min_pixels: int, max_pixels: max):
    return list(filter(lambda x: min_pixels < cv2.contourArea(x) < max_pixels, contours))
