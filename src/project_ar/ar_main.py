import cv2
import numpy as np

from project_ar.obj_model import OBJ
from project_ar import qr_detector
from project_ar.trackbar_generator import TrackbarSettings


def __bin_imag__(img, threshold, morph_struct_size):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret2, bin_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)

    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_struct_size, morph_struct_size))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_ERODE, structure)

    cv2.imshow('Settings', bin_img)
    return bin_img


DEFAULT_COLOR = (0, 255, 255)


def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


def render(img, obj, projection_matrix, target_height, target_width, color=False, obj_scale=1):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = target_height, target_width

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points *= obj_scale
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection_matrix)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img


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
    l = np.math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / np.math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / np.math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)


if __name__ == '__main__':
    main_window_name: str = 'Webcam'

    cap = cv2.VideoCapture(0)

    # Binary Image Settings
    threshold = 127
    threshold_trackbar = TrackbarSettings('Settings')
    morph_trackbar = TrackbarSettings('Settings', 'Morph Size Struct', 2, 30)

    qr_image = cv2.imread("../assets/QR_1.png")
    # Opencv QR Decoder
    qrDecoder = cv2.QRCodeDetector()
    qr_height, qr_width, _ = qr_image.shape
    qr_bin_frame = __bin_imag__(qr_image, threshold, 1)
    # Opencv QR Decoder main result for homography
    original_bbox = qr_detector.__detect_bbox__(qrDecoder, bin_frame=qr_bin_frame)

    # Obj to render
    obj_dir = "../assets/models/fox.obj"
    obj = OBJ(obj_dir, swapyz=True)
    obj_scale = 1
    # Obj scale render
    obj_scale_trackbar = TrackbarSettings(main_window_name, 'Scale', 1, 5)

    camera_parameters = [[539.5314058058052, 0.0, 213.90895733662572],
                         [0.0, 543.5813662750779, 275.7850229389913],
                         [0.0, 0.0, 1.0]]

    while True:
        if cv2.waitKey(1) == ord('q'): break

        # Capture frame video
        ret, cam_frame = cap.read()
        cam_frame = cv2.flip(cam_frame, 1)

        threshold = threshold_trackbar.update()
        morph_struct_size = morph_trackbar.update()
        morph_struct_size = morph_struct_size if morph_struct_size > 1 else 1
        bin_frame = __bin_imag__(cam_frame, threshold, morph_struct_size)

        # Detect QR bbox from frame
        bbox = qr_detector.__detect_bbox__(qrDecoder, bin_frame=bin_frame, cam_frame_draw=cam_frame)

        if bbox is not None:
            src_pts = np.float32(original_bbox)
            dst_pts = np.float32(bbox)
            # Find homography between original QR image and target/destination image
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # obtain 3D projection matrix from homography matrix and camera parameters
            projection = projection_matrix(camera_parameters, homography)

            if homography is not None:
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = projection_matrix(camera_parameters, homography)
                    obj_scale = obj_scale_trackbar.update()
                    cam_frame = render(cam_frame, obj, projection, qr_height, qr_width, False, obj_scale)
                except:
                    pass

        cv2.imshow(main_window_name, cam_frame)