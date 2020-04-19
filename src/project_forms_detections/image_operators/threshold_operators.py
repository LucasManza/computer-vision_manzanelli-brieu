import cv2


def generate_binary_image(image, threshold: int, invert_image: bool = False):
    """"

    """

    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh_type = cv2.THRESH_BINARY if invert_image else cv2.THRESH_BINARY_INV

    ret, binary_img = cv2.threshold(gray_frame, threshold, 255, thresh_type)

    return binary_img
