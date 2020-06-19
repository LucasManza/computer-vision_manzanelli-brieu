import cv2 as cv
import numpy as np

from project_robotic_vision.calibration import camera_calibration

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    camera_calibration.load_intrinsic_params('camera_intrinsic_params.json')
    undistortion: bool = False

    while True:

        if cv.waitKey(1) == ord('q'):
            break

        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        cv.imshow('WebCam', frame)

        calibrate_img = camera_calibration.calibrate_image(frame)
        cv.imshow('Calibrate Image', calibrate_img)

cv.destroyAllWindows()
