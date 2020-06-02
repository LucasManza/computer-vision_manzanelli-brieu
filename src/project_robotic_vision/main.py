import cv2

from practice.homography_practice.homography_tool import HomographyTool

homographyTool = HomographyTool()
web_cam_img_name = "WebCam"

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # draw circle here (etc...)
        homographyTool.add_point(x, y)
        print('x = %d, y = %d' % (x, y))


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    cv2.namedWindow(web_cam_img_name)
    cv2.setMouseCallback(web_cam_img_name, on_click)

    while True:
        if cv2.waitKey(1) == ord('q'):
            break

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        show_img = frame

        if homographyTool.__2DPoints__.__len__() == 4:
            homo_img = homographyTool.four_point_transform(frame)
            # todo missing fixing camera calibration

            cv2.imshow('Homography', homo_img)
            show_img = homographyTool.draw_points(frame)

        cv2.imshow(web_cam_img_name, show_img)

cv2.destroyAllWindows()
