import json
import math
import cv2 as cv
import numpy as np
import yaml

camera_matrix = None
distortion_coeff = None


def load_intrinsic_params():
    with open('camera_intrinsic_params.json') as json_file:
        data = json.load(json_file)
        global camera_matrix
        camera_matrix = np.array(data['camera_matrix'])
        print('-- Camera Matrix --')
        print(camera_matrix)
        global distortion_coeff
        distortion_coeff = np.array(data['distortion_coeff'])
        print('-- Distortion Coeff --')
        print(distortion_coeff)


def calibrate_image(camera_img):
    # Get camera image shape
    h, w = camera_img.shape[:2]

    # # Get fixed camera matrix and roi
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w, h), 1, (w, h))
    #
    # # Fixed undistortion
    dst = cv.undistort(camera_img, camera_matrix, distortion_coeff, None, new_camera_matrix)

    return dst


def __calibrate_camera__(images, chessboard_size: (int, int) = (6, 7)):
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
    print("-- Camera Matrix --")
    print(camera_matrix)
    print("-- Distortion Coeff. --")
    print(distortion_coeff)

    __save_as_json__(camera_matrix, distortion_coeff)

    print("-- Calibration Finished! --")


def __save_as_json__(cam_matrix, dist_coeff):
    print("-- Writing JSON  --")
    cam_matrix = cam_matrix.tolist()
    dist_coeff = dist_coeff.tolist()
    data = {"distortion_coeff": dist_coeff, "camera_matrix": cam_matrix}
    with open("camera_intrinsic_params.json", "w") as f:
        json.dump(data, f)


def __capture_cam_frame__(dst_path: str, amount=50) -> list:
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
            print('Saving image ' + str(count))
            capturing = count % 5 == 0

    print('---CAPTURE IMAGE---')
    return images_capture


def __load_images_from_folder__(folder):
    print('--- Loading images--')
    images = []
    for filename in cv.os.listdir(folder):
        img = cv.imread(cv.os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def __write_line_instruction__(line, main_img, line_numb, spacing):
    # Write some Text
    font = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (30, 30 + line_numb * spacing)
    fontScale = 0.6
    fontColor = (255, 255, 255)
    lineType = 2

    cv.putText(main_img, line,
               bottomLeftCornerOfText,
               font,
               fontScale,
               fontColor,
               lineType)


if __name__ == '__main__':
    instruction_img = np.zeros((300, 700, 3), np.uint8)

    lines: list = [
        'Camera Intrinsic Params',
        '1A) For starting capture chessboard images: Press \'s\'',
        '1B) Capture chessboard image by: Pressing \'c\'',
        '2) For start calibrate camera by images captured: Press \'c\'',
        '3) Exit: Press \'q\'',
        'Note: Pay Attention to your terminal response.',
    ]
    for num, line in enumerate(lines, start=1):
        __write_line_instruction__(line, instruction_img, num, 40)

    cv.imshow('Camera Intrinsic Params', instruction_img)

    while True:
        if cv.waitKey(1) == ord('q'):
            break

        elif cv.waitKey(1) == ord('s'):
            __capture_cam_frame__('caps', 50)

        elif cv.waitKey(1) == ord('c'):
            images = __load_images_from_folder__('caps')
            __calibrate_camera__(images)

    cv.destroyAllWindows()
