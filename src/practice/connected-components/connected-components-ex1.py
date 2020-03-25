import cv2
import numpy as np


# todo
#    Ejercicios con la cámara, en tiempo real, mostrando siempre en una ventana la imagen de la cámara.  Requiere imágenes binarias sin ruido, que se pueden obtener por thresholding y operaciones morfológicas con sus respectivos trackbars.
#   1)Aplicar componentes conectados a la imagen binaria de la cámara, y visualizarla coloreando con un color diferente cada grupo, usando colorMap

def generate_binary_image(image, threshold: int):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY_INV)

    return binary


def reduce_noise(binary_img, structure_size):
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structure_size, structure_size))
    open_result = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, structure)
    closed_result = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, structure)

    return closed_result


def show_connected_components(binary_image):
    # num_labels, labels = cv2.connectedComponents(binary_image)
    # img_int = np.uint8(labels)
    #
    # color_map_img = cv2.applyColorMap(img_int, cv2.COLORMAP_HSV)
    # cv2.imshow('component', color_map_img)

    num_labels, labels = cv2.connectedComponents(binary_image)
    binaryImageClone = np.copy(labels)

    # Find the max and min pixel values and their locations
    (minValue, maxValue, minPosition, maxPosition) = cv2.minMaxLoc(binaryImageClone)

    # normalize the image so that the min value is 0 and max value is 255
    binaryImageClone = 255 * (binaryImageClone - minValue) / (maxValue - minValue)

    # convert image to 8bits unsigned type
    binaryImageClone = np.uint8(binaryImageClone)

    # Apply a color map
    img_color_map = cv2.applyColorMap(binaryImageClone, cv2.COLORMAP_JET)
    cv2.imshow('component', img_color_map)


def nothing(x):
    pass


if __name__ == '__main__':
    origin_img = cv2.imread('../../assets/numbers.png')
    threshold = 127
    dilation_struct_size = 1

    cv2.namedWindow("Window", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Threshold", "Window", threshold, 255, nothing)
    cv2.createTrackbar("Erode Structure Size", "Window", dilation_struct_size, 50, nothing)

    while True:

        if cv2.waitKey(1) == ord('q'):
            break

        # Capture frame-by-frame

        threshold = cv2.getTrackbarPos("Threshold", "Window")
        dilation_struct_size = cv2.getTrackbarPos("Erode Structure Size", "Window")
        dilation_struct_size = 1 if dilation_struct_size < 2 else dilation_struct_size

        # Display the resulting frame
        cv2.imshow('Ants', origin_img)

        bin_img = generate_binary_image(origin_img, threshold)

        bin_img = reduce_noise(bin_img, dilation_struct_size)
        cv2.imshow('Binary Image', bin_img)

        show_connected_components(bin_img)

    cv2.destroyAllWindows()
