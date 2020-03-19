import cv2


# todo Ejercicios con la cámara, en tiempo real, mostrando siempre en una ventana la imagen de la cámara.
#  Requiere imágenes binarias, que se pueden obtener por thresholding.
#  2) Erosionar

# Generate binary image.
def generate_binary_image(image, threshold: int):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)

    return binary


def reduce_noise(binary_img, structure_size):
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structure_size, structure_size))
    result = binary_img
    for x in range(100):
        open_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, structure)
        result = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, structure)
    cv2.imshow('reduce image image', result)


def nothing(x):
    pass


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

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
        cv2.imshow('frame', frame)

        bin_img = generate_binary_image(frame, threshold)

        reduce_noise(bin_img, dilation_struct_size)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
