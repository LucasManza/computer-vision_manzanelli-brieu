# todo  3) Generar una imagen binaria normal sobre la cámara, usando ambos métodos de umbral adaptativo
#  , controlando el tamaño de bloque con una barra deslizante


import cv2


# Show normal, binary and binary inverted images. With automatic OTSU and triangle thresh settings.
def show_binary_images(image):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary_adapt_media = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    binary_adapt_gaussian = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                  11, 2)

    cv2.imshow('original', gray_frame)
    cv2.imshow('binary + media', binary_adapt_media)
    cv2.imshow('binary + gaussian', binary_adapt_gaussian)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while True:

        if cv2.waitKey(1) == ord('q'):
            break

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        show_binary_images(frame)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
