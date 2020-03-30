import cv2
import numpy as np


# todo
#    Ejercicios con la cámara, en tiempo real, mostrando siempre en una ventana la imagen de la cámara.  Requiere imágenes binarias sin ruido, que se pueden obtener por thresholding y operaciones morfológicas con sus respectivos trackbars.
#   1)Aplicar componentes conectados a la imagen binaria de la cámara, y visualizarla coloreando con un color diferente cada grupo, usando colorMap

def generate_binary_image(image, threshold: int):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)

    return binary


def reduce_noise(binary_img, structure_size):
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structure_size, structure_size))
    open_result = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, structure)
    closed_result = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, structure)

    return closed_result


def show_connected_components(binary_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
    binaryImageClone = np.copy(labels)

    # Find the max and min pixel values and their locations
    (minValue, maxValue, minPosition, maxPosition) = cv2.minMaxLoc(binaryImageClone)

    # normalize the image so that the min value is 0 and max value is 255
    binaryImageClone = 255 * (binaryImageClone - minValue) / (maxValue - minValue)

    # convert image to 8bits unsigned type
    binaryImageClone = np.uint8(binaryImageClone)

    # Apply a color map
    img_color_map = cv2.applyColorMap(binaryImageClone, cv2.COLORMAP_JET)

    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)

    # print(stats[labels[2], cv2.CC_STAT_AREA][0])
    # print(centroids[labels[2]][0])

    for label in labels:
        ### Get area from a matrix of same value
        pixels = stats[label, cv2.CC_STAT_AREA][0]
        print(pixels)
        ### Exercise condition request
        if 100 < pixels < 10000:
            ### Get centroid from a matrix of same value
            centroid = centroids[label][0]
            top_left_point = (int(centroid[0] - 100), int(centroid[1] + 100))
            bottom_right_point = (int(centroid[0] + 100), int(centroid[1] - 100))
            img_color_map = cv2.rectangle(img_color_map, top_left_point, bottom_right_point, red_color, 2)

    cv2.imshow('component', img_color_map)


def nothing(x):
    pass


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # origin_img = cv2.imread('../../assets/numbers.png')
    threshold = 127
    dilation_struct_size = 1

    cv2.namedWindow("Window", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Threshold", "Window", threshold, 255, nothing)
    cv2.createTrackbar("Erode Structure Size", "Window", dilation_struct_size, 50, nothing)

    while True:

        if cv2.waitKey(1) == ord('q'):
            break

        # Capture frame-by-frame
        ret, frame = cap.read()

        threshold = cv2.getTrackbarPos("Threshold", "Window")
        dilation_struct_size = cv2.getTrackbarPos("Erode Structure Size", "Window")
        dilation_struct_size = 1 if dilation_struct_size < 2 else dilation_struct_size

        # Display the resulting frame
        cv2.imshow('Img', frame)

        bin_img = generate_binary_image(frame, threshold)

        bin_img = reduce_noise(bin_img, dilation_struct_size)
        cv2.imshow('Binary Image', bin_img)

        show_connected_components(bin_img)

    cv2.destroyAllWindows()
