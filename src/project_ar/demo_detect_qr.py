import cv2
import numpy as np

from project_ar.trackbar_generator import TrackbarSettings

GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
YELLOW_COLOR = (0, 255, 255)


def __draw_coord__(image, coord, color):
    cv2.circle(image, coord, radius=0, color=color, thickness=10)


def __bin_imag__(img, threshold, morph_struct_size):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret2, bin_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)

    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_struct_size, morph_struct_size))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_ERODE, structure)

    cv2.imshow('Binary Webcam', bin_img)
    return bin_img


def __nothing__(self, x):
    pass


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    main_window_name: str = 'Webcam'
    threshold = 127

    threshold_trackbar = TrackbarSettings('Settings')
    morph_trackbar = TrackbarSettings('Settings', 'Morph Size Struct', 2, 30)

    qrDecoder = cv2.QRCodeDetector()

    while True:
        if cv2.waitKey(1) == ord('q'):
            break

        ret, cam_frame = cap.read()
        cam_frame = cv2.flip(cam_frame, 1)

        threshold = threshold_trackbar.update()
        morph_struct_size = morph_trackbar.update()
        morph_struct_size = morph_struct_size if morph_struct_size > 1 else 1
        bin_frame = __bin_imag__(cam_frame, threshold, morph_struct_size)

        detected, points = qrDecoder.detect(bin_frame)

        if detected and points is not None:
            n_lines = len(points)

            top_right = tuple(points[0][0])
            bottom_right = tuple(points[1][0])
            bottom_left = tuple(points[2][0])
            top_left = tuple(points[3][0])

            __draw_coord__(cam_frame, top_right, RED_COLOR)
            __draw_coord__(cam_frame, bottom_right, YELLOW_COLOR)
            __draw_coord__(cam_frame, bottom_left, GREEN_COLOR)
            __draw_coord__(cam_frame, top_left, BLUE_COLOR)



        cv2.imshow('Webcam', cam_frame)
