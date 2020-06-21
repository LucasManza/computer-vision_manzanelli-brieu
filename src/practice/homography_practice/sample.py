import cv2

from practice.homography_practice.homography_tool import HomographyTool

homographyTool = HomographyTool()


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # draw circle here (etc...)
        homographyTool.add_point(x,y)
        print('x = %d, y = %d' % (x, y))



if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("WebCam")
    cv2.setMouseCallback('WebCam', on_click)


    while True:

        if cv2.waitKey(1) == ord('q'):
            break

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if homographyTool.__2DPoints__.__len__() == 4:
            cv2.imshow('Homography', homographyTool.rect_homography(frame))
        cv2.imshow('WebCam', frame)

cv2.destroyAllWindows()
