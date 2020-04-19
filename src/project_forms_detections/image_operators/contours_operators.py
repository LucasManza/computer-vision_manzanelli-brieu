import cv2


def find_contours(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours(img, contours, colour):
    copy_img = img.copy()
    cv2.drawContours(copy_img, contours, -1, colour, 2)
    return copy_img


# Apply a rectangle detection by contours, instead of drawing the contour per se.
def draw_contours_rect(img, contours, colour):
    copy_img = img.copy()
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        cv2.rectangle(copy_img, (x, y), (x + w, y + h), colour, 2)
    return copy_img
