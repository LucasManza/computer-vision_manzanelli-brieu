import cv2 as cv
import numpy as np

from project_robotic_vision.calibration import camera_calibration

if __name__ == '__main__':
    CHESSBOARD_SIZE = (6, 7)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.


    cap = cv.VideoCapture(0)

    while True:

        if cv.waitKey(1) == ord('q'):
            break

        ret, frame = cap.read()
        frame = cv.flip(frame, 1)

        if cv.waitKey(1) == ord('c'):
            camera_calibration.calibrate(frame)


        cv.imshow('WebCam', frame)

cv.destroyAllWindows()
