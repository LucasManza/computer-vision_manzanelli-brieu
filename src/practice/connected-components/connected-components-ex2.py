import cv2
import numpy as np


def apply_color_map(img):
    img_color_map = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imshow('component', img_color_map)


def nothing(x):
    pass


if __name__ == '__main__':
    origin_img = cv2.imread('../../assets/numbers.png')
    threshold = 127
    dilation_struct_size = 1

    cv2.namedWindow("Window", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Threshold", "Window", threshold, 255, nothing)
    cv2.createTrackbar("Erode Structure Size", "Window", dilation_struct_size, 50, nothing)

    while True:

        if cv2.waitKey(1) == ord('q'):
            break

        # Capture frame-by-frame

        threshold = cv2.getTrackbarPos("Threshold", "Window")
        dilation_struct_size = cv2.getTrackbarPos("Erode Structure Size", "Window")
        dilation_struct_size = 1 if dilation_struct_size < 2 else dilation_struct_size

        # Display the resulting frame
        cv2.imshow('Ants', origin_img)

        apply_color_map(origin_img)

    cv2.destroyAllWindows()
