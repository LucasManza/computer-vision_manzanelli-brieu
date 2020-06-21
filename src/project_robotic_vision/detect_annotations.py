import cv2

from project_forms_detections.image_operators import contours_operators


def __points_to_rect_coords__(top_left: (int, int), bottom_right: (int, int)):
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    rect_coords = top_left, top_right, bottom_right, bottom_left
    center_x = ((bottom_right[0] - top_left[0]) / 2) + top_left[0]
    center_y = ((bottom_right[1] - top_left[1]) / 2) + top_left[1]
    center_coord = int(center_x), int(center_y)
    return rect_coords, center_coord


def draw_system_ref(image, color):
    h, w, c = image.shape
    rect, center = __points_to_rect_coords__((0, 0), (w, h))
    (tl, tr, br, bl) = rect

    thickness = 2
    cv2.rectangle(image, tl, center, color, thickness)
    cv2.rectangle(image, center, br, color, thickness)
    cv2.rectangle(image, tl, br, color, thickness)


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
    cv2.putText(img, text, (x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)


def __parse_text_annotation__(center: (int, int), image, centimeters: int) -> str:
    unit: str = 'px'
    h, w = image.shape[:2]
    x, y = center
    x -= w / 2
    y -= h / 2
    if centimeters is not None:
        unit = 'cm'
        x = (center[0] / w * centimeters - centimeters / 2)
        y = (center[1] / h * centimeters - centimeters / 2)
        x = round(x,2)
        y = round(y,2)

    y *= -1
    return '(' + str(x) + unit + ', ' + str(y) + unit + ')'


def __apply_homo_matrix__(homo_matrix, matrix2D):
    matrix3D = matrix2D[0], matrix2D[1], 1
    # Apply homo. matrix transformation over center
    matrix3D = homo_matrix.dot(matrix3D)
    return int(matrix3D[0]), int(matrix3D[1])


def draw_annotations(img, contours, color, homo_matrix=None, centimeters: int = None):
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
        if homo_matrix is not None:
            center = __apply_homo_matrix__(homo_matrix, center)
            x, y = __apply_homo_matrix__(homo_matrix, (x, y))

        # Draw center
        cv2.circle(copy_img, center, color=color, radius=0, thickness=5)
        # Draw rectangle container with text
        cv2.rectangle(copy_img, (x, y), (x + w, y + h), (36, 255, 12), 1)
        txt = __parse_text_annotation__(center, img, centimeters)
        __text_annotation__(copy_img, txt, (x, y), color)

    return copy_img
