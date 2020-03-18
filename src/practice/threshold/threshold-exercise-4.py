# todo  4) Generar una imagen binaria con dos umbrales con inRange,
#  para segmentar un objeto por su color en el espacio HSV

import cv2
import numpy as np

min_hue = 0
min_sat = 0
min_value = 0
max_hue = 180
max_sat = 180
max_value = 255


# Show normal, binary and binary inverted images. With automatic OTSU and triangle thresh settings.
def show_binary_images(image, size: int):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary_adapt_media = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                               size, 1)
    binary_adapt_gaussian = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                  size, 1)  # size only odd values ?? what's 1?

    cv2.imshow('original', image)
    cv2.imshow('binary + media', binary_adapt_media)
    cv2.imshow('binary + gaussian', binary_adapt_gaussian)


def nothing(x):
    pass


def create_trackbars():
    cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Block Size%", "Trackbars", 3, 100, nothing)
    cv2.createTrackbar("Max. H", "Trackbars", max_value, 180, nothing)
    cv2.createTrackbar("Max. S", "Trackbars", max_sat, 255, nothing)
    cv2.createTrackbar("Max. V", "Trackbars", max_value, 255, nothing)
    cv2.createTrackbar("Min. H", "Trackbars", min_hue, 180, nothing)
    cv2.createTrackbar("Min. S", "Trackbars", min_sat, 255, nothing)
    cv2.createTrackbar("Min. V", "Trackbars", min_value, 255, nothing)


def get_block_size():
    block_size_result = cv2.getTrackbarPos("Block Size%", "Trackbars")
    return odd_block_size(block_size_result)


def get_min_hsv():
    return (cv2.getTrackbarPos("Min. H", "Trackbars"),
            cv2.getTrackbarPos("Min. S", "Trackbars"),
            cv2.getTrackbarPos("Min. V", "Trackbars"))


def get_max_hsv():
    return (cv2.getTrackbarPos("Max. H", "Trackbars"),
            cv2.getTrackbarPos("Max. S", "Trackbars"),
            cv2.getTrackbarPos("Max. V", "Trackbars"))


def odd_block_size(value):
    value = int(np.ceil(value))
    if value < 2: return 3
    return value + 1 if value % 2 == 0 else value


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    block_size = 3

    create_trackbars()

    while True:
        if cv2.waitKey(1) == ord('q'):
            break

        # Capture frame-by-frame
        ret, frame = camera.read()

        block_size = get_block_size()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, get_min_hsv(), get_max_hsv())

        frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Display the resulting frames
        show_binary_images(frame, block_size)

    # When everything done, release the capture
    camera.release()
    cv2.destroyAllWindows()
