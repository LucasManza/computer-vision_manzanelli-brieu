import cv2
import math

red_color = (0, 0, 255)
blue_color = (255, 0, 0)
green_color = (0, 255, 0)


def __show_text__(original_img, labels: list):
    font = cv2.FONT_HERSHEY_SIMPLEX
    offset = 35
    x, y = 50, 50
    for idx, label in enumerate(labels):
        cv2.putText(original_img, label, (x, y + offset * idx), font,
                    0.5, green_color, 2, cv2.LINE_AA)


def __generate_binary_image__(image, threshold: int, invert: bool):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    tresh_code = cv2.THRESH_BINARY if not invert else cv2.THRESH_BINARY_INV
    ret, binary = cv2.threshold(gray_image, threshold, 255, tresh_code)

    return binary


def __reduce_noise__(binary_img, structure_size):
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structure_size, structure_size))
    open_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, structure)
    result = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, structure)
    return result


def __draw_contours_img__(original_img, contours):
    cv2.drawContours(original_img, contours, -1, red_color, 2)


def __find_contours__(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def __show_moments_data__(binary_img, original_img):
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

    __show_text__(original_img, [
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

    return huMoments


def __print_moments_data__(binary_images):
    all_hu_moments: list = []

    for bin_img in binary_images:
        moments = cv2.moments(bin_img)
        hu_moments = cv2.HuMoments(moments)
        all_hu_moments.append(hu_moments)

    print('----HO----')
    for hu_moments in all_hu_moments:
        h0 = hu_moments[0]
        print(h0)
    print('----H1----')
    for hu_moments in all_hu_moments:
        h1 = hu_moments[1]
        print(h1)
    print('----H2----')
    for hu_moments in all_hu_moments:
        h2 = hu_moments[2]
        print(h2)
    print('----H3----')
    for hu_moments in all_hu_moments:
        h3 = hu_moments[3]
        print(h3)
    print('----H4----')
    for hu_moments in all_hu_moments:
        h4 = hu_moments[4]
        print(h4)
    print('----H5----')
    for hu_moments in all_hu_moments:
        h5 = hu_moments[5]
        print(h5)
    print('----H6----')
    for hu_moments in all_hu_moments:
        h6 = hu_moments[6]
        print(h6)


def analyze(image,
            image_name: str,
            invert_bin_img: bool,
            threshold: int,
            dilation_struct_size):
    # The following code generate binary img, reduce noise and show binary image
    bin_img = __generate_binary_image__(image, threshold, invert_bin_img)
    bin_img = __reduce_noise__(bin_img, dilation_struct_size)
    # cv2.imshow('binary_' + image_name, bin_img)

    # Find contours
    contours = __find_contours__(bin_img)

    # Draw contours in red and show moments data
    __draw_contours_img__(image, contours)
    __show_moments_data__(bin_img, image)
    cv2.imshow('contours_img' + image_name, image)


def compare_images_hu(images,
                      images_name: str,
                      invert_bin_img: bool,
                      threshold: int,
                      dilation_struct_size):
    bin_images: list = []
    # The following code generate binary img, reduce noise and show binary image
    for image in images:
        bin_img = __generate_binary_image__(image, threshold, invert_bin_img)
        bin_img = __reduce_noise__(bin_img, dilation_struct_size)
        bin_images.append(bin_img)

    __print_moments_data__(bin_images)
