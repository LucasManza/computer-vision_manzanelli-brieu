import cv2

from practice.homography_practice.homography_tool import HomographyTool, points_to_rect_coords
from project_forms_detections.image_settings import ImageSettings
from project_robotic_vision import detector
from project_robotic_vision.calibration import camera_calibration

homographyTool = HomographyTool()
webcam_window_name = "WebCam"
homo_size_cm = 15


def __nothing__(x):
    pass


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # draw circle here (etc...)
        homographyTool.add_point(x, y)
        print('x = %d, y = %d' % (x, y))


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    camera_calibration.load_intrinsic_params('../project_robotic_vision/calibration/camera_intrinsic_params.json')

    cv2.namedWindow(webcam_window_name)
    cv2.setMouseCallback(webcam_window_name, on_click)

    img_target = cv2.imread('../../src/assets/Capture.PNG')

    camera_settings = ImageSettings('Camera Analyzer Window', morph_struct_size=4)
    target_settings = ImageSettings('Target Analyzer Window')

    cv2.createTrackbar("cm/px", webcam_window_name, homo_size_cm, 200, __nothing__)

    show_binary_images = True

    camera_invert_img: bool = False
    target_invert_img: bool = False

    while True:
        if cv2.waitKey(1) == ord('q'):
            break

        ret, cam_frame = cap.read()
        cam_frame = camera_calibration.calibrate_image(cam_frame)
        cam_frame = cv2.flip(cam_frame, 1)

        if homographyTool.__2DPoints__.__len__() == 4:
            homo_img, homo_matrix = homographyTool.rect_homography(cam_frame)

            cam_frame, homo_img = detector.detector_target(
                cam_frame,
                homo_img,
                homo_matrix,
                homo_size_cm,
                img_target, camera_settings, target_settings,
                target_invert_img,
                camera_invert_img,
                show_binary_images)

            homo_size_cm = cv2.getTrackbarPos("cm/px", webcam_window_name)
            cv2.imshow('Homography', homo_img)
            homographyTool.draw_points(cam_frame)

        cv2.imshow(webcam_window_name, cam_frame)

cv2.destroyAllWindows()
