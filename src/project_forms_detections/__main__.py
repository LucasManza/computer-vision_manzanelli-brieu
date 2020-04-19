import cv2

from project_forms_detections.image_operators import threshold_operators as threshold_operators
from project_forms_detections.image_operators import morphological_operators as morph_operators


def nothing(x):
    pass


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    threshold = 127
    morph_struct_size = 10

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

        # Generate monochromatic image
        bin_img = threshold_operators.generate_binary_image(frame, threshold)
        
        # Clean monochromatic by reduce noise using dilation and closing morph operators
        bin_clean_img = morph_operators.reduce_noise_dil_closing(bin_img, morph_struct_size)

        cv2.imshow('analyze_img', bin_clean_img)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
