import math

import cv2 as cv
import numpy as np
import yaml


def calibrate_image(camera_img):
    # Get camera image shape
    h, w = camera_img.shape[:2]

    # Get fixed camera matrix and roi
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w, h), 1, (w, h))

    # Fixed undistortion
    dst = cv.undistort(camera_img, camera_matrix, distortion_coeff, None, new_camera_matrix)

    cv.imshow('Undistortion', dst)

    return dst


def calibrate_camera(images, chessboard_size: (int, int) = (6, 7)):
    print('--- Calibration in Proccess--')
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    for img in images:
        # img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)

    # Get camera calibration params
    ret, camera_matrix, distortion_coeff, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                                                            None, None)

    camera_intrinsic_params = dict(
        ret=ret,
        camera_matrix=camera_matrix,
        distortion_coeff=distortion_coeff,
        rvecs=rvecs,
        tvecs=tvecs,
    )
    with open('camera_intrinsic_params.yml', 'w') as outfile:
        yaml.dump(camera_intrinsic_params, outfile, default_flow_style=False)


def capture_cam_frame(dst_path: str, amount=50) -> list:
    print('--- START CAPTURE IMAGES ---')
    print('Press C continually for capturing images')
    cap = cv.VideoCapture(0)
    frameRate = cap.get(60)  # frame rate
    count: int = 0
    images_capture: list = []
    capturing: bool = False

    while count <= amount:
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()

        if not ret:
            break

        if cv.waitKey(1) == ord('c'):
            capturing = True

        if capturing and frameId % math.floor(frameRate) == 0:
            cv.imwrite(dst_path + '/cap_c' + str(count) + '.png', frame)
            count = count + 1
            print('Saving image ' + count)
            capturing = count % 5 == 0

    print('---CAPTURE IMAGE---')
    return images_capture


def load_images_from_folder(folder):
    print('--- Loading images--')
    images = []
    for filename in cv.os.listdir(folder):
        img = cv.imread(cv.os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


if __name__ == '__main__':
    img = np.zeros((512, 512, 3), np.uint8)

    # Write some Text
    text = '1) For start capturing images: Press s\n2) For start calibrate camera by images capture: Press c\n3) Exit: Press q'
    font = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0, 100)
    fontScale = 0.3
    fontColor = (255, 255, 255)
    lineType = 2

    cv.putText(img, text,
               bottomLeftCornerOfText,
               font,
               fontScale,
               fontColor,
               lineType)

    cv.imshow('Main', img)

    while True:
        if cv.waitKey(1) == ord('q'):
            break

        elif cv.waitKey(1) == ord('s'):
            capture_cam_frame('caps', 50)

        elif cv.waitKey(1) == ord('c'):
            images = load_images_from_folder('caps')
            calibrate_camera(images)

    cv.destroyAllWindows()
