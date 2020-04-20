from enum import Enum

import cv2

from project_forms_detections.colours.rgb.colours import green_colour, red_colour
from src.project_forms_detections.image_analyzer import ImageAnalyzer
from project_forms_detections.image_operators import contours_operators as contours_operators


class ImgToShowEnum(Enum):
    ORIGINAL: 1
    BINARY: 2
    CONTOURS: 3


def __next_to_show__(current: ImgToShowEnum):
    if current == ImgToShowEnum.ORIGINAL:
        return ImgToShowEnum.BINARY
    elif current == ImgToShowEnum.BINARY:
        return ImgToShowEnum.CONTOURS
    else:
        return ImgToShowEnum.ORIGINAL


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # target_window_name: str = 'Target Window'
    img_target = cv2.imread('../assets/star.png')

    camera_analyzer = ImageAnalyzer('Camera Analyzer Window')
    target_analyzer = ImageAnalyzer('Target Analyzer Window')

    # camera_img_show: ImgToShowEnum = ImgToShowEnum.ORIGINAL
    # target_img_show: ImgToShowEnum = ImgToShowEnum.ORIGINAL

    camera_invert_img: bool = False
    target_invert_img: bool = False

    while True:
        if cv2.waitKey(1) == ord('q'):
            break

        if cv2.waitKey(1) == ord('i'):
            camera_invert_img = not camera_invert_img

        if cv2.waitKey(1) == ord('j'):
            target_invert_img = not target_invert_img

        # Capture frame-by-frame
        ret, frame = cap.read()

        (target_bin_image, target_contours) = target_analyzer.analyze_image(img_target, invert_image=target_invert_img)

        img_target_contours = contours_operators.draw_contours(img_target, target_contours, green_colour)
        target_analyzer.update(img_target_contours)

        (camera_bin_image, camera_contours) = camera_analyzer.analyze_image(frame, invert_image=camera_invert_img)
        camera_analyzer.update(camera_bin_image)

        contours_result = contours_operators.filter_contours_by_match_contours(camera_contours, target_contours[0],
                                                                               0.01)

        if contours_result.__len__() > 0:
            img_detection = contours_operators.draw_contours_rect(frame, contours_result, green_colour)
            cv2.imshow('Shape Detection', img_detection)
        else:
            cv2.imshow('Shape Detection', frame)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
