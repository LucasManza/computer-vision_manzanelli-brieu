import cv2

from practice.homography_practice.homography_tool import HomographyTool
from project_forms_detections.image_settings import ImageSettings
from project_robotic_vision import vision_detector
from project_robotic_vision.calibration import camera_calibration

homographyTool = HomographyTool()
web_cam_img_name = "WebCam"


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # draw circle here (etc...)
        homographyTool.add_point(x, y)
        print('x = %d, y = %d' % (x, y))


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    camera_calibration.load_intrinsic_params('../project_robotic_vision/calibration/camera_intrinsic_params.json')

    cv2.namedWindow(web_cam_img_name)
    cv2.setMouseCallback(web_cam_img_name, on_click)

    img_target = cv2.imread('../../src/assets/circle-img.png')

    camera_settings = ImageSettings('Camera Analyzer Window', morph_struct_size=4)
    target_settings = ImageSettings('Target Analyzer Window')

    show_binary_images = True

    camera_invert_img: bool = False
    target_invert_img: bool = False

    while True:
        if cv2.waitKey(1) == ord('q'):
            break

        ret, cam_frame = cap.read()
        cam_frame = camera_calibration.calibrate_image(cam_frame)
        cam_frame = cv2.flip(cam_frame, 1)
        show_img = cam_frame

        if homographyTool.__2DPoints__.__len__() == 2:
            homo_img = homographyTool.rect_homography(cam_frame)
            homo_img = vision_detector.detect_shape(
                homo_img, img_target, camera_settings, target_settings,
                target_invert_img,
                camera_invert_img,
                show_binary_images)

            cv2.imshow('Homography', homo_img)
            show_img = homographyTool.draw_points(cam_frame)

        cv2.imshow(web_cam_img_name, show_img)

cv2.destroyAllWindows()
