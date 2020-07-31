import cv2
import numpy as np

# def draw_system_ref(image, color):
#     h, w, c = image.shape
#     rect, center = __points_to_rect_coords__((0, 0), (w, h))
#     (tl, tr, br, bl) = rect
#
#     thickness = 2
#     cv2.rectangle(image, tl, center, color, thickness)
#     cv2.rectangle(image, center, br, color, thickness)
#     cv2.rectangle(image, tl, br, color, thickness)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    # camera_calibration.load_intrinsic_params('../project_robotic_vision/calibration/camera_intrinsic_params.json')

    qrDecoder = cv2.QRCodeDetector()

    while True:
        if cv2.waitKey(1) == ord('q'):
            break

        ret, cam_frame = cap.read()
        cam_frame = cv2.flip(cam_frame, 1)

        gray_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)

        detected, points = qrDecoder.detect(cam_frame)

        if detected and points is not None:
            # display the image with lines
            # length of bounding box
            n_lines = len(points)
            for i in range(n_lines):
                # draw all lines
                point1 = tuple(points[i][0])
                point2 = tuple(points[(i + 1) % n_lines][0])
                cv2.line(cam_frame, point1, point2, color=(255, 0, 0), thickness=2)

        cv2.imshow('Webcam', cam_frame)
