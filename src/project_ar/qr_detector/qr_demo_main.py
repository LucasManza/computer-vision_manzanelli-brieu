import cv2
import numpy as np

from project_ar.obj_model import OBJ
from project_ar.qr_detector import qr_detector
from project_ar.trackbar_generator import TrackbarSettings


def __bin_imag__(img, threshold, morph_struct_size):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret2, bin_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)

    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_struct_size, morph_struct_size))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_ERODE, structure)

    cv2.imshow('Settings', bin_img)
    return bin_img


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def rect_homography(image, rect):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(rect))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (maxWidth, maxHeight))

    # return the warped image
    return warped, transform_matrix


DEFAULT_COLOR = (0, 0, 0)


def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


def render(img, obj, projection, target_height, target_width, color=False):
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
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
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
    cap = cv2.VideoCapture(0)

    threshold = 127
    main_window_name: str = 'Webcam'

    threshold_trackbar = TrackbarSettings('Settings')
    morph_trackbar = TrackbarSettings('Settings', 'Morph Size Struct', 2, 30)

    qrDecoder = cv2.QRCodeDetector()
    obj_dir = "../../assets/pirate-ship-fat.obj"
    obj = OBJ(obj_dir, swapyz=True)

    camera_parameters = [[539.5314058058052, 0.0, 213.90895733662572],
                         [0.0, 543.5813662750779, 275.7850229389913],
                         [0.0, 0.0, 1.0]]

    qr_image = cv2.imread("../../assets/QR_1.png")
    h, w, _ = qr_image.shape

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
        bbox = qr_detector.__detect_bbox__(qrDecoder, cam_frame, bin_frame)

        if bbox is not None:

            warped_img, h_matrix = rect_homography(cam_frame, bbox)

            # obtain 3D projection matrix from homography matrix and camera parameters
            projection = projection_matrix(camera_parameters, h_matrix)

            # if h_matrix is not None:
            #     try:
            #         # obtain 3D projection matrix from homography matrix and camera parameters
            #         projection = projection_matrix(camera_parameters, h_matrix)
            #
            #         cam_frame = render(cam_frame, obj, projection, h, w, False)
            #         # frame = render(frame, model, projection)
            #     except:
            #         pass

        cv2.imshow(main_window_name, cam_frame)
