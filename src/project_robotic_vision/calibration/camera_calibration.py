import cv2 as cv
import numpy as np


def calibrate(camera_img, chessboard_size: (int, int) = (6, 7)):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    gray = cv.cvtColor(camera_img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
    # If found, add object points, image points (after refining them)
    if ret == False: return

    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners)
    # Draw and display the corners
    cv.drawChessboardCorners(camera_img, chessboard_size, corners2, ret)
    cv.imshow('Detection_Calibration', camera_img)

    # Get camera calibration params
    ret, camera_matrix, distortion_coeff, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                                                            None, None)
    # Get camera image shape
    h, w = camera_img.shape[:2]

    # Get fixed camera matrix and roi
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w, h), 1, (w, h))

    # Fixed undistortion
    dst = cv.undistort(camera_img, camera_matrix, distortion_coeff, None, new_camera_matrix)

    cv.imshow('Undistortion', dst)

    return dst
