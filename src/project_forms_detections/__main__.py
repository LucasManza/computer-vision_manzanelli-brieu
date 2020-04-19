import cv2

# SHOW erosion image by using Ellipsis structure. Args must be de binary image and structure size
from project_forms_detections.image_operators.morphological_operators import erode
from project_forms_detections.image_operators.threshold_operators import generate_binary_image


def show_erosion_image(binary_img, structure_size):
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structure_size, structure_size))
    erode_img = cv2.morphologyEx(binary_img, cv2.MORPH_ERODE, structure)
    cv2.imshow('erode image', erode_img)


def nothing(x):
    pass


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    threshold = 127
    morph_struct_size = 1

    cv2.namedWindow("Window", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Threshold", "Window", threshold, 255, nothing)
    cv2.createTrackbar("Structure Size", "Window", morph_struct_size, 50, nothing)

    while True:

        if cv2.waitKey(1) == ord('q'):
            break

        # Capture frame-by-frame
        ret, frame = cap.read()

        threshold = cv2.getTrackbarPos("Threshold", "Window")
        morph_struct_size = cv2.getTrackbarPos("Structure Size", "Window")
        morph_struct_size = 1 if morph_struct_size < 2 else morph_struct_size

        # Display the resulting frame
        cv2.imshow('frame', frame)

        bin_img = generate_binary_image(frame, threshold)
        bin_img = erode(bin_img, morph_struct_size)
        cv2.imshow('analyze_img', bin_img)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
