# Generate monochromatic image for Target Image
import cv2

from project_forms_detections.colours.rgb.colours import red_colour, green_colour
from project_forms_detections.image_operators import threshold_operators, contours_operators
from project_forms_detections.image_operators.contours_operators import DetectionContourEnum
from project_forms_detections.image_operators import morphological_operators as  morph_operators


def __show_contours__(image, contours):
    return contours_operators.draw_contours(image, contours, red_colour)


def __select_img__(select_bin: bool, img, bin_img, contours):
    if select_bin:
        return bin_img
    else:
        return __show_contours__(img, contours)


def __text_annotation__(img, text: str, size: (int, int), color):
    x, y = size

    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    # set the text start position
    text_offset_y = y - 10
    # make the coords of the box with a small padding of two pixels
    box_coords = ((x, text_offset_y), (x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(img, box_coords[0], box_coords[1], color, cv2.FILLED)
    cv2.putText(img, text, (x, text_offset_y), font, fontScale=font_scale, color=(255, 255, 255), thickness=1)


def __apply_homo_matrix__(homo_matrix, matrix2D):
    matrix3D = matrix2D[0], matrix2D[1], 1
    # Apply homo. matrix transformation over center
    matrix3D = homo_matrix.dot(matrix3D)
    return int(matrix3D[0]), int(matrix3D[1])


def __draw_annotations__(img, contours, color, transf_matrix=None):
    """"
    Apply a rectangle detection by contours, instead of drawing the contour per se.
    """
    copy_img = img.copy()
    for cont in contours:
        # Contours rect
        x, y, w, h = cv2.boundingRect(cont)
        # Get center
        center = contours_operators.contour_center(cont)

        # Apply Transformation
        if transf_matrix is not None:
            center = __apply_homo_matrix__(transf_matrix, center)
            # h, w = __apply_homo_matrix__(transf_matrix, (h, w))
            x, y = __apply_homo_matrix__(transf_matrix, (x, y))

        # Draw center
        cv2.circle(copy_img, center, color=color, radius=0, thickness=5)
        # Draw rectangle container with text
        cv2.rectangle(copy_img, (x, y), (x + w, y + h), (36, 255, 12), 1)
        __text_annotation__(copy_img, str(center), (x, y), color)

    return copy_img


def __show_shapes_detection__(camera_frame, contours_result, transf_matrix):
    img_detection = camera_frame
    if contours_result.__len__() > 0:
        img_detection = __draw_annotations__(camera_frame, contours_result, green_colour, transf_matrix)
    return img_detection


def __points_to_rect_coords__(top_left: (int, int), bottom_right: (int, int)):
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    rect_coords = top_left, top_right, bottom_right, bottom_left
    center_x = ((bottom_right[0] - top_left[0]) / 2) + top_left[0]
    center_y = ((bottom_right[1] - top_left[1]) / 2) + top_left[1]
    center_coord = int(center_x), int(center_y)
    return rect_coords, center_coord


def __draw_system_ref__(image, color):
    h, w, c = image.shape
    rect, center = __points_to_rect_coords__((0, 0), (w, h))
    (tl, tr, br, bl) = rect

    thickness = 2
    cv2.rectangle(image, tl, center, color, thickness)
    cv2.rectangle(image, center, br, color, thickness)
    cv2.rectangle(image, tl, br, color, thickness)


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
    camera_image = __show_shapes_detection__(camera_image, contours_result, None)
    homo_image = __show_shapes_detection__(homo_image, contours_result, homo_matrix)
    __draw_system_ref__(camera_image, (255, 0, 0))

    return camera_image, homo_image
