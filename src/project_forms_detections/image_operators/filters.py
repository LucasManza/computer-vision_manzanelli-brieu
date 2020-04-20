import cv2


def __distance__(contour, target_contour):
    return cv2.matchShapes(contour, target_contour, cv2.CONTOURS_MATCH_I3, 0)


def contours_distance(contours, target_contours, match_error: float):
    if target_contours is None or target_contours.__len__() == 0: return []

    return list(filter(lambda x: __distance__(x, target_contours) <= match_error, contours))


def contours_area(contours, min_pixels: int, max_pixels: max):
    return list(filter(lambda x: min_pixels < cv2.contourArea(x) < max_pixels, contours))
