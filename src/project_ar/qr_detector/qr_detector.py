import cv2

GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
YELLOW_COLOR = (0, 255, 255)


def __draw_coord__(image, coord, color):
    cv2.circle(image, coord, radius=0, color=color, thickness=10)


def __draw_bbox__(cam_frame, top_right, bottom_right, bottom_left, top_left):
    __draw_coord__(cam_frame, top_right, RED_COLOR)
    __draw_coord__(cam_frame, bottom_right, RED_COLOR)
    __draw_coord__(cam_frame, bottom_left, GREEN_COLOR)
    __draw_coord__(cam_frame, top_left, RED_COLOR)


def __detect_bbox__(qrDecoder, cam_frame, bin_frame):
    # Detect QR bbox from frame
    data, bbox, rectifiedImage = qrDecoder.detectAndDecode(bin_frame)

    if bbox is not None:
        top_right = tuple(bbox[0][0])
        bottom_right = tuple(bbox[1][0])
        bottom_left = tuple(bbox[2][0])
        top_left = tuple(bbox[3][0])
        __draw_bbox__(cam_frame, top_right, bottom_right, bottom_left, top_left)
        return top_right, bottom_right, bottom_left, top_left
    else:
        return None
