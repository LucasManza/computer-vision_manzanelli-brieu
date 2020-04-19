import cv2

from project_forms_detections.image_operators import threshold_operators as threshold_operators
from project_forms_detections.image_operators import morphological_operators as morph_operators
from project_forms_detections.image_operators import contours_operators as contours_operators
from project_forms_detections.colours.rgb.colours import blue_colour, green_colour, red_colour

analyze_window_name: str = 'Analyze Window'


def nothing(x):
    pass


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    threshold = 127
    morph_struct_size = 10
    approx_poly_dp = 1

    cv2.namedWindow(analyze_window_name, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Threshold", analyze_window_name, threshold, 255, nothing)
    cv2.createTrackbar("Structure Size", analyze_window_name, morph_struct_size, 50, nothing)
    cv2.createTrackbar("AproxPolyDP", "Window", approx_poly_dp, 50, nothing)

    while True:

        if cv2.waitKey(1) == ord('q'):
            break

        # Capture frame-by-frame
        ret, frame = cap.read()

        threshold = cv2.getTrackbarPos("Threshold", analyze_window_name)
        morph_struct_size = cv2.getTrackbarPos("Structure Size", analyze_window_name)
        morph_struct_size = 1 if morph_struct_size < 2 else morph_struct_size

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Generate monochromatic image
        bin_img = threshold_operators.generate_binary_image(frame, threshold)

        # Clean monochromatic by reduce noise using dilation and closing morph operators
        bin_clean_img = morph_operators.reduce_noise_dil_closing(bin_img, morph_struct_size)

        # Find contours over a binary image (clean)
        contours = contours_operators.find_contours(bin_clean_img)

        contours_img = contours_operators.draw_contours_rect(frame, contours, red_colour)

        cv2.imshow(analyze_window_name, contours_img)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
