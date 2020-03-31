import cv2

red_color = (0, 0, 255)
blue_color = (255, 0, 0)


def generate_binary_image(image, threshold: int):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)

    return binary


def reduce_noise(binary_img, structure_size):
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structure_size, structure_size))
    open_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, structure)
    result = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, structure)
    return result


def draw_contours_img(original_img, binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(original_img, contours, -1, red_color, 2)


def draw_contours_img_by_approx_poly(original_img, binary_img, approx_poly):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result_img = original_img

    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, approx_poly, True)
        result_img = cv2.drawContours(original_img, [approx], -1, blue_color, 2)

    return result_img


def nothing(x):
    pass


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    threshold = 127
    dilation_struct_size = 1
    approx_poly_dp = 1

    cv2.namedWindow("Window", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Threshold", "Window", threshold, 255, nothing)
    cv2.createTrackbar("Erode Structure Size", "Window", dilation_struct_size, 50, nothing)
    cv2.createTrackbar("AproxPolyDP", "Window", approx_poly_dp, 50, nothing)

    while True:

        if cv2.waitKey(1) == ord('q'):
            break

        # Capture frame-by-frame
        ret, frame = cap.read()

        threshold = cv2.getTrackbarPos("Threshold", "Window")
        dilation_struct_size = cv2.getTrackbarPos("Erode Structure Size", "Window")
        dilation_struct_size = 1 if dilation_struct_size < 2 else dilation_struct_size

        # Display the resulting frame
        cv2.imshow('frame', frame)

        bin_img = generate_binary_image(frame, threshold)

        bin_img = reduce_noise(bin_img, dilation_struct_size)
        cv2.imshow('binary', bin_img)
        draw_img = draw_contours_img(frame, bin_img)
        cv2.imshow('contours aprox poly', draw_contours_img_by_approx_poly(draw_img, bin_img, approx_poly_dp))

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
