# Generate monochromatic image for Target Image
import cv2

from project_forms_detections.colours.rgb.colours import red_colour, green_colour
from project_forms_detections.image_operators import threshold_operators, contours_operators
from project_forms_detections.image_operators.contours_operators import DetectionContourEnum
from project_forms_detections.image_operators import morphological_operators as  morph_operators
from project_robotic_vision.detect_annotations import draw_system_ref, draw_annotations


def __show_contours__(image, contours):
    return contours_operators.draw_contours(image, contours, red_colour)


def __select_img__(select_bin: bool, img, bin_img, contours):
    if select_bin:
        return bin_img
    else:
        return __show_contours__(img, contours)


def __show_shapes_detection__(camera_frame, contours_result, transf_matrix, centimeters):
    img_detection = camera_frame
    if contours_result.__len__() > 0:
        img_detection = draw_annotations(camera_frame, contours_result, green_colour, transf_matrix, centimeters)
    return img_detection


def detector_target(
        camera_image,
        homo_image,
        homo_matrix,
        img_target,
        camera_settings,
        target_settings,
        target_invert_img,
        camera_invert_img,
        show_binary_images,
):
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
    contours_result = contours_operators.filter_by_area(contours_result, 100, 10000)

    # It's show a new window all possible results
    camera_image = __show_shapes_detection__(camera_image, contours_result, transf_matrix=None, centimeters=None)
    homo_image = __show_shapes_detection__(homo_image, contours_result, transf_matrix=homo_matrix, centimeters=15)
    draw_system_ref(camera_image, (255, 0, 0))
    draw_system_ref(homo_image, (255, 0, 0))

    return camera_image, homo_image
