import math
import os

import cv2
import numpy as np

from project_ar.obj_model import OBJ
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


def render(img, obj, projection, model, color=False):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w, c = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img


def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


def projection_matrix(camera_parameters, homography):
    """
     From the camera calibration matrix and the estimated homography
     compute the 3D projection matrix
     """

    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    main_window_name: str = 'Webcam'

    obj = OBJ("../../src/assets/99-intergalactic_spaceship-obj/Intergalactic_Spaceship-(Wavefront).obj",
              swapyz=True)

    qr_target = cv2.imread('../../src/assets/QR_1.png')
    camera_parameters = np.array([
        [
            539.5314058058052,
            0.0,
            213.90895733662572
        ],
        [
            0.0,
            543.5813662750779,
            275.7850229389913
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ])

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

        MIN = 30

        if len(match_descriptors) > MIN:
            # draw first 15 matches.
            det_frame = cv2.drawMatches(qr_target, key_points_target, cam_frame, key_points_frame,
                                        match_descriptors[:MIN], 0, flags=2)
            cv2.imshow('Detect', det_frame)

            src_pts = np.float32([key_points_target[m.queryIdx].pt for m in match_descriptors]).reshape(-1, 1, 2)
            dst_pts = np.float32([key_points_frame[m.trainIdx].pt for m in match_descriptors]).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Draw a rectangle that marks the found model in the frame
            h, w, c = qr_target.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # project corners into frame
            dst = cv2.perspectiveTransform(pts, homography)
            # connect them with lines
            frame = cv2.polylines(cam_frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            # if a valid homography matrix was found render cube on model plane
            if homography is not None:
                # obtain 3D projection matrix from homography matrix and camera parameters
                projection = projection_matrix(camera_parameters, homography)
                # project cube or model
                frame = render(frame, obj, projection, qr_target, False)
                # frame = render(frame, model, projection)
            cv2.imshow('frame', frame)

        cv2.imshow('Webcam', cam_frame)
