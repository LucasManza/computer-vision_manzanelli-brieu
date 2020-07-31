import cv2
import numpy as np

from project_ar.trackbar_generator import TrackbarSettings

GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
YELLOW_COLOR = (0, 255, 255)


def __draw_coord__(image, coord, color):
    cv2.circle(image, coord, radius=0, color=color, thickness=10)


def __bin_imag__(img, threshold, morph_struct_size):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret2, bin_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)

    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_struct_size, morph_struct_size))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_ERODE, structure)

    cv2.imshow('Binary Webcam', bin_img)
    return bin_img


def __match_descriptors__(descriptors_target, descriptors_frame, distance_criteria: float):
    bf = cv2.BFMatcher()
    keypoints_matches = bf.knnMatch(descriptors_target, descriptors_frame, k=2)
    result = []
    # keypoints_results = list(map(
    #     lambda kp_t, kp_d: kp_t.distance < distance_criteria * kp_d.distance,
    #     keypoints_matches))

    for kp_i, kp_j in keypoints_matches:
        if kp_i.distance < distance_criteria * kp_j.distance:
            result.append(kp_j)

    return result


def __nothing__(self, x):
    pass


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    main_window_name: str = 'Webcam'
    threshold = 127

    threshold_trackbar = TrackbarSettings('Settings')
    morph_trackbar = TrackbarSettings('Settings', 'Morph Size Struct', 2, 30)

    qr_target = cv2.imread('../../src/assets/QR_1.png')

    # percent by which the image is resized
    scale_percent = 30

    # calculate the 50 percent of original dimensions
    width = int(qr_target.shape[1] * scale_percent / 100)
    height = int(qr_target.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    qr_target = cv2.resize(qr_target, dsize)

    height_target, width_target, c = qr_target.shape

    orb = cv2.ORB_create(nfeatures=1000)
    # Generate keypoints and descriptors for QR image
    key_points_target, descriptors_target = orb.detectAndCompute(qr_target, None)

    while True:
        if cv2.waitKey(1) == ord('q'):
            break

        ret, cam_frame = cap.read()
        cam_frame = cv2.flip(cam_frame, 1)
        cam_frame = cv2.resize(cam_frame, (width_target, height_target))

        # Generate keypoints and descriptors for frame
        key_points_frame, descriptors_frame = orb.detectAndCompute(cam_frame, None)

        match_descriptors = __match_descriptors__(descriptors_target, descriptors_frame, 0.75)

        # frame_features = cv2.drawMatches(qr_target, key_points_target, cam_frame, key_points_frame, match_descriptors,
        #                                  None, flags=2)

        if len(match_descriptors) > 20:
            src_pts = np.float32([key_points_target[m.queryIdx].pt for m in match_descriptors]).reshape(-1, 1, 2)
            dst_pts = np.float32([key_points_frame[m.trainIdx].pt for m in match_descriptors]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            print(matrix)

            pts = np.float32([[0, 0], [0, height_target], [width_target, height_target], [width_target, 0]])
            pts = pts.reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            cam_frame = cv2.polylines(cam_frame, [np.int32(dst)], True, (255, 0, 255), 3, cv2.LINE_AA)  # draw polylines

        cv2.imshow('Webcam', cam_frame)
