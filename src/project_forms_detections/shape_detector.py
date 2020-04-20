from enum import Enum

import cv2

from project_forms_detections.colours.rgb.colours import green_colour, red_colour
from src.project_forms_detections.image_analyzer import ImageAnalyzer
from project_forms_detections.image_operators import contours_operators as contours_operators


def __show_contours__(image, contours):
    return contours_operators.draw_contours(image, contours, red_colour)


def __select_img__(select_bin: bool, img, bin_img, contours):
    if select_bin:
        return bin_img
    else:
        return __show_contours__(img, contours)


def __show_shapes_detection__(camera_frame, contours_result):
    if contours_result.__len__() > 0:
        img_detection = contours_operators \
            .draw_contours_rect(camera_frame, contours_result, green_colour)
        cv2.imshow('Shape Detection', img_detection)
    else:
        cv2.imshow('Shape Detection', camera_frame)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    img_target = cv2.imread('../assets/star.png')

    camera_analyzer = ImageAnalyzer('Camera Analyzer Window')
    target_analyzer = ImageAnalyzer('Target Analyzer Window')

    show_binary_images = True

    camera_invert_img: bool = False
    target_invert_img: bool = False

    while True:
        if cv2.waitKey(1) == ord('q'):
            break

        if cv2.waitKey(1) == ord('i'):
            camera_invert_img = not camera_invert_img

        if cv2.waitKey(1) == ord('j'):
            target_invert_img = not target_invert_img

        if cv2.waitKey(1) == ord('n'):
            show_binary_images = not show_binary_images

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Generate for TARGET Image, bin image and contours
        (bin_img_target, contours_target) = target_analyzer.analyze_image(img_target, invert_image=target_invert_img)

        target_analyzer.update(__select_img__(show_binary_images, img_target, bin_img_target, contours_target))

        (bin_img_camera, contours_camera) = camera_analyzer.analyze_image(frame, invert_image=camera_invert_img)

        camera_analyzer.update(__select_img__(show_binary_images, frame, bin_img_camera, contours_camera))

        contours_result = contours_operators \
            .filter_contours_by_match_contours(contours_camera, contours_target[0], 0.01)

        __show_shapes_detection__(frame, contours_result)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
