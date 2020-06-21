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


class HomographyTool:

    def __init__(self):
        self.__2DPoints__ = []
        self.__rect_coords__ = []

    def __points_to_rect_coords__(self):
        top_left_coord = self.__2DPoints__[0]
        bottom_right_coord = self.__2DPoints__[1]
        top_right_coord = (bottom_right_coord[0], top_left_coord[1])
        bottom_left_coord = (top_left_coord[0], bottom_right_coord[1])

        self.__rect_coords__ = top_left_coord, top_right_coord, bottom_right_coord, bottom_left_coord
        
    def add_point(self, x, y):
        if len(self.__2DPoints__) == 1:
            self.__2DPoints__.append((x, y))
            self.__points_to_rect_coords__()
        else:
            self.__2DPoints__ = []
            self.__rect_coords__ = []
            self.__2DPoints__.append((x, y))

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
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        # return the warped image
        return warped

    def draw_points(self, camera_img):
        image = camera_img
        for point in self.__rect_coords__:
            image = cv2.circle(camera_img, point, radius=0, color=(0, 0, 255), thickness=5)
        return image

        # for point in self.__2DPoints__:
        #     camera_img[point[0], point[1]] = [0, 0, 255]
