from enum import Enum

import cv2

from project_forms_detections.colours.rgb.colours import green_colour, red_colour
from project_forms_detections.image_operators.contours_operators import DetectionContourEnum
from src.project_forms_detections.image_settings import ImageSettings
from project_forms_detections.image_operators import threshold_operators as  threshold_operators
from project_forms_detections.image_operators import morphological_operators as  morph_operators
from project_forms_detections.image_operators import contours_operators as  contours_operators


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

    camera_settings = ImageSettings('Camera Analyzer Window', morph_struct_size=4)
    target_settings = ImageSettings('Target Analyzer Window')

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
        ret, camera_image = cap.read()

        # Generate monochromatic image for Target Image
        target_binary_image = threshold_operators \
            .generate_binary_image(img_target, target_settings.threshold, target_invert_img)

        # Generate monochromatic image for Camera Image
        camera_binary_image = threshold_operators \
            .generate_binary_image(camera_image, camera_settings.threshold, camera_invert_img)

        # Clean binary target image by erosion
        target_binary_image = morph_operators. \
            erode(target_binary_image, target_settings.morph_struct_size)

        # Clean binary camera image by closing reduce noise
        camera_binary_image = morph_operators \
            .reduce_noise_erode_opening(camera_binary_image, camera_settings.morph_struct_size)

        # Target contours
        target_contours = contours_operators.find_contours(target_binary_image,
                                                           DetectionContourEnum.EXTERNAL_CONT_DETECT)

        # Camera contours
        camera_contours = contours_operators.find_contours(camera_binary_image)

        # Update settings methods and show image option for target
        target_settings.update(__select_img__(show_binary_images, img_target, target_binary_image, target_contours))

        # Idem for camera
        camera_settings.update(__select_img__(show_binary_images, camera_image, camera_binary_image, camera_contours))

        # Match contour detection with target, by filtering camera contours
        contours_result = contours_operators.filter_by_distance(camera_contours, target_contours[0], 0.01)

        # Filter outliers contours with a specific min and max amount of area pixels
        contours_result = contours_operators.filter_by_area(contours_result, 2000, 10000)

        # It's show a new window all possible results
        __show_shapes_detection__(camera_image, contours_result)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
