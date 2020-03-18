# todo  3) Generar una imagen binaria normal sobre la cámara, usando ambos métodos de umbral adaptativo
#  , controlando el tamaño de bloque con una barra deslizante


import cv2
import numpy as np


# Show normal, binary and binary inverted images. With automatic OTSU and triangle thresh settings.
def show_binary_images(image, size: int):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary_adapt_media = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                               size, 1)
    binary_adapt_gaussian = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                  size, 1)  # size only odd values ?? what's 1?

    cv2.imshow('binary + media', binary_adapt_media)
    cv2.imshow('binary + gaussian', binary_adapt_gaussian)


def nothing(x):
    print(x)
    pass


def set_block_size(value):
    value = int(np.ceil(value))
    if value < 2: return 3
    return value + 1 if value % 2 == 0 else value


if __name__ == '__main__':
    block_size = 3
    camera = cv2.VideoCapture(0)

    cv2.namedWindow("Block Size", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Block Size%", "Block Size", 3, 100, nothing)

    while True:
        if cv2.waitKey(1) == ord('q'):
            break

        # Capture frame-by-frame
        ret, frame = camera.read()

        result = cv2.getTrackbarPos("Block Size%", "Block Size")
        block_size = set_block_size(result)

        # Display the resulting frames
        show_binary_images(frame, block_size)

    # When everything done, release the capture
    camera.release()
    cv2.destroyAllWindows()
