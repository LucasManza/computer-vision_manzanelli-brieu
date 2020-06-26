import cv2
import numpy as np


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def points_to_rect_coords(top_left: (int, int), top_right: (int, int), bottom_right: (int, int),
                          bottom_left: (int, int)):
    # top_right = (bottom_right[0], top_left[1])
    # bottom_left = (top_left[0], bottom_right[1])

    rect_coords = top_left, top_right, bottom_right, bottom_left
    center_x = ((bottom_right[0] - top_left[0]) / 2) + top_left[0]
    center_y = ((bottom_right[1] - top_left[1]) / 2) + top_left[1]
    center_coord = int(center_x), int(center_y)
    return rect_coords, center_coord


class HomographyTool:

    def __init__(self):
        self.__2DPoints__ = []
        self.__rect_coords__ = []
        self.__center_coord__ = []

    def add_point(self, x, y):
        if len(self.__2DPoints__) == 4:
            self.__2DPoints__ = []
            self.__rect_coords__ = []
            self.__2DPoints__.append((x, y))
            print(len(self.__2DPoints__))
        else:
            self.__2DPoints__.append((x, y))
            print(len(self.__2DPoints__))

            if len(self.__2DPoints__) == 4:
                self.__rect_coords__, self.__center_coord__ = \
                    points_to_rect_coords(
                        top_left=self.__2DPoints__[0],
                        top_right=self.__2DPoints__[1],
                        bottom_right=self.__2DPoints__[2],
                        bottom_left=self.__2DPoints__[3],
                    )

    def rect_homography(self, image):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = order_points(np.array(self.__rect_coords__))
        (tl, tr, br, bl) = rect
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        # compute the perspective transform matrix and then apply it
        transform_matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, transform_matrix, (maxWidth, maxHeight))
        # return the warped image
        return warped, transform_matrix
