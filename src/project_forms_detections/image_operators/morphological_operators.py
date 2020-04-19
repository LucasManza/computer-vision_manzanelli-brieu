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


def __opening__(binary_img, structure_size):
    morph_structure = __generate_morph_structure__(structure_size)
    return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, morph_structure)


def __closing__(binary_img, structure_size):
    morph_structure = __generate_morph_structure__(structure_size)
    return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, morph_structure)


def reduce_noise_dil_closing(binary_img, structure_size):
    dil_img = dilation(binary_img, structure_size)
    return __closing__(dil_img, structure_size)


def reduce_noise_erode_opening(binary_img, structure_size):
    erode_img = erode(binary_img, structure_size)
    return __opening__(erode_img, structure_size)
