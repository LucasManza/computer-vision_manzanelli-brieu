import cv2
import math

red_color = (0, 0, 255)
blue_color = (255, 0, 0)
green_color = (0, 255, 0)


def show_text(original_img, labels: list):
    font = cv2.FONT_HERSHEY_SIMPLEX
    offset = 35
    x, y = 50, 50
    for idx, label in enumerate(labels):
        cv2.putText(original_img, label, (x, y + offset * idx), font,
                    0.5, green_color, 2, cv2.LINE_AA)


def generate_binary_image(image, threshold: int, invert: bool):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    tresh_code = cv2.THRESH_BINARY if not invert else cv2.THRESH_BINARY_INV
    ret, binary = cv2.threshold(gray_frame, threshold, 255, tresh_code)

    return binary


def reduce_noise(binary_img, structure_size):
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structure_size, structure_size))
    open_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, structure)
    result = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, structure)
    return result


def draw_contours_img(original_img, contours):
    cv2.drawContours(original_img, contours, -1, red_color, 2)


# Apply a rectangle detection by contours, instead of drawing the contour per se.
def draw_contours_by_bounding_react(img, contours):
    img_result = img
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        cv2.rectangle(img, (x, y), (x + w, y + h), green_color, 2)
    return img_result


def find_contours(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def show_moments_data(binary_img, original_img):
    moments = cv2.moments(binary_img)
    huMoments = cv2.HuMoments(moments)

    m00 = moments['m00']
    m10 = moments['m10']
    m01 = moments['m01']

    h0 = huMoments[0]
    h1 = huMoments[1]
    h2 = huMoments[2]
    h3 = huMoments[3]
    h4 = huMoments[4]
    h5 = huMoments[5]
    h6 = huMoments[6]

    show_text(original_img, [
        'h0: ' + str(h0),
        'h1: ' + str(h1),
        'h2: ' + str(h2),
        'h3: ' + str(h3),
        'h4: ' + str(h4),
        'h5: ' + str(h5),
        'h6: ' + str(h6),
    ])

    if m00 != 0:
        x, y = (m10 / m00, m01 / m00)
        radius = math.sqrt(m00)
        cv2.circle(original_img, (int(x), int(y)), int(radius), green_color, 3)


def nothing(x):
    pass


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    invert_bin_img: bool = False
    threshold = 127
    dilation_struct_size = 1

    cv2.namedWindow("Window", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Threshold", "Window", threshold, 255, nothing)
    cv2.createTrackbar("Erode Structure Size", "Window", dilation_struct_size, 50, nothing)

    while True:

        if cv2.waitKey(1) == ord('q'):
            break

        # ret, frame = cap.read()

        # frame = cv2.imread('../../assets/start.png')
        frame = cv2.imread('../../assets/dataset/start/2.png')

        if cv2.waitKey(1) == ord('i'):
            invert_bin_img = not invert_bin_img

        threshold = cv2.getTrackbarPos("Threshold", "Window")
        dilation_struct_size = cv2.getTrackbarPos("Erode Structure Size", "Window")
        dilation_struct_size = 1 if dilation_struct_size < 2 else dilation_struct_size

        # The following code generate binary img, reduce noise and show binary image
        bin_img = generate_binary_image(frame, threshold, invert_bin_img)
        bin_img = reduce_noise(bin_img, dilation_struct_size)
        cv2.imshow('binary', bin_img)

        # Find contours
        contours = find_contours(bin_img)

        # Draw contours in red and show moments data
        draw_contours_img(frame, contours)
        show_moments_data(bin_img, frame)
        cv2.imshow('contours_img', frame)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
