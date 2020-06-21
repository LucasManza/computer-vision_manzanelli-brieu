import json
import math
import cv2 as cv
import numpy as np
import yaml

# Auxiliaries variables for calibrate camera function
camera_matrix = None
distortion_coeff = None

# Internal Variables
# Chessboard square cell size in meters.  __SQUARE_SIZE__ = 1.5 cm
__SQUARE_SIZE__ = 0.015

# Amount of picture for capture process
__AMOUNT_OF_IMG__ = 20

SUCCESS = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'

def load_intrinsic_params(path: str):
    """"Load camera intrinsic parameters which has been saved as JSON.
   :param path: The path where the file was saved.
   :type path: str

    """
    with open(path) as json_file:
        data = json.load(json_file)
        global camera_matrix
        camera_matrix = np.array(data['camera_matrix'])
        print('--- Camera Matrix ---')
        print(camera_matrix)
        global distortion_coeff
        distortion_coeff = np.array(data['distortion_coeff'])
        print('--- Distortion Coeff ---')
        print(distortion_coeff)


def calibrate_image(image):
    """"Calibrate image by using matrix_camera and distortion coefficients.

    It's required to use load_intrinsic_params function first for calibration image.

    :param image: The original image that you want to be calibrate. You use camera frame too.
    :type image: OpenCV image

    :return:The Calibrate image. If any error has been occurred, a message appears and the original image is return.

    """
    if camera_matrix is None or distortion_coeff is None:
        print('YOU MUST LOAD PARAMETERS OR GENERATE THEM FIRST')
        return image

    # Get camera image shape
    h, w = image.shape[:2]

    # # Get fixed camera matrix and roi
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w, h), 1, (w, h))

    #  Fixed undistortion
    dst = cv.undistort(image, camera_matrix, distortion_coeff, None, new_camera_matrix)

    return dst


def __print_error__(error):
    if error < 0.02:
        print(SUCCESS + "Total error: "+str(error)+ENDC)
    elif 0.02 <= error < 0.03:
        print(WARNING + "Total error: "+str(error)+ENDC)
    else:
        print(FAIL + "Total error: "+str(error)+ENDC)


def __calculate_params_process__(images, square_size: float, chessboard_size: (int, int) = (7, 6)):
    print('--- Calibration in Proccess ---')
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

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
            cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv.imshow('Calibrate Window', img)
            cv.waitKey(500)

    # Get camera calibration params
    ret, camera_matrix, distortion_coeff, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                                                            None, None)
    print("-- Camera Matrix --")
    print(camera_matrix)
    print("-- Distortion Coeff. --")
    print(distortion_coeff)

    __save_params__(camera_matrix, distortion_coeff)

    tot_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, distortion_coeff)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        tot_error += error

    __print_error__(tot_error / len(objpoints))

    print("-- Calibration Finished! --")


def __save_params__(cam_matrix, dist_coeff):
    print("-- Writing JSON  --")
    cam_matrix = cam_matrix.tolist()
    dist_coeff = dist_coeff.tolist()
    data = {"distortion_coeff": dist_coeff, "camera_matrix": cam_matrix}
    with open("camera_intrinsic_params.json", "w") as f:
        json.dump(data, f)


def __capture_images_process__(dst_path: str, amount=50) -> list:
    print('--- START CAPTURE IMAGES ---')

    print('Press C continually for   capturing images')
    cap = cv.VideoCapture(0)
    window_name: str = 'Webcam'
    frameRate = cap.get(60)  # frame rate
    count: int = 0
    images_capture: list = []
    capturing: bool = False

    while count <= amount:
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        cv.imshow(window_name, frame)

        if not ret:
            break

        if cv.waitKey(1) == ord('c'):
            capturing = True

        if capturing and frameId % math.floor(frameRate) == 0:
            cv.imwrite(dst_path + '/cap_c' + str(count) + '.png', frame)
            count = count + 1
            print('Saving image ' + str(count))
            capturing = count % 5 == 0

    print('--- FINISHED CAPTURE IMAGE PROCESS ---')
    cv.destroyWindow(window_name)
    cap.release()
    return images_capture


def __load_capture_images__(folder):
    print('--- Loading images ---')
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


def __show_calibration_results__():
    cap = cv.VideoCapture(0)
    load_intrinsic_params('camera_intrinsic_params.json')
    original_window_name: str = 'Webcam'
    calibrate_window_name: str = 'Calibrate Window'

    while True:

        if cv.waitKey(1) == ord('i'):
            break

        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        cv.imshow(original_window_name, frame)

        calibrate_img = calibrate_image(frame)
        cv.imshow(calibrate_window_name, calibrate_img)

    cv.destroyWindow(original_window_name)
    cv.destroyWindow(calibrate_window_name)
    cap.release()


if __name__ == '__main__':
    instruction_img = np.zeros((550, 900, 3), np.uint8)

    lines: list = [
        'Steps:',
        '1) Start by capture ' + str(__AMOUNT_OF_IMG__) + ' Chessboard Images:',
        '  * First Press: \'s\', for initialized the process',
        '  * Then continually Press:\'c\' for capture each image',
        '  * Await for the terminal to notified you that the process has been finished',
        '2) Start calibration camera process by Pressing: \'c\'. Await for terminal response.',
        '3) Show calibration results',
        '  * Turn on video by Pressing: \'u\'',
        '  * Turn off video by Press: \'i\'',
        '4) Exit: Press \'q\'',
        'Note: Pay Attention to your terminal response.',
    ]
    for num, line in enumerate(lines, start=1):
        __write_line_instruction__(line, instruction_img, num, 40)

    cv.imshow('Camera Intrinsic Params', instruction_img)

    while True:
        if cv.waitKey(0) == ord('q'):
            break

        elif cv.waitKey(0) == ord('s'):
            __capture_images_process__('caps', __AMOUNT_OF_IMG__)

        elif cv.waitKey(0) == ord('c'):
            images = __load_capture_images__('caps')
            __calculate_params_process__(images, __SQUARE_SIZE__)

        elif cv.waitKey(0) == ord('u'):
            __show_calibration_results__()

    cv.destroyAllWindows()
