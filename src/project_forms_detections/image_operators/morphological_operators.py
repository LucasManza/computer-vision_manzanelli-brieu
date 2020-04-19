import cv2


def __generate_morph_structure__(structure_size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structure_size, structure_size))


def dilation(binary_img, structure_size):
    morph_structure = __generate_morph_structure__(structure_size)
    return cv2.morphologyEx(binary_img, cv2.MORPH_DILATE, morph_structure)


def erode(binary_img, structure_size):
    morph_structure = __generate_morph_structure__(structure_size)
    erode_img = cv2.morphologyEx(binary_img, cv2.MORPH_ERODE, morph_structure)
    return erode_img


def opening(binary_img, structure_size):
    morph_structure = __generate_morph_structure__(structure_size)
    return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, morph_structure)


def closing(binary_img, structure_size):
    morph_structure = __generate_morph_structure__(structure_size)
    return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, morph_structure)
