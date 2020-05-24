import numpy as np
import cv2 as cv
import glob

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

        img = frame

        if cv.waitKey(1) == ord('c'):
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
                cv.imshow('calibration', img)

        cv.imshow('img', img)

cv.destroyAllWindows()
