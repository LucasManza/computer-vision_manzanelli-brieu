# todo  1) Generar una imagen binaria normal y
#  otra invertida sobre la cámara, controlando el umbral con una barra deslizante


import cv2


# Show normal, binary and binary inverted images.
def show_binary_images(image, threshold: int):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    ret, binary_inv = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow('original', gray_frame)
    cv2.imshow('binary', binary)
    cv2.imshow('binary_inv', binary_inv)


def nothing(x):
    pass


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    threshold = 127

    cv2.namedWindow("Window", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Threshold", "Window", 127, 255, nothing)

    while True:

        if cv2.waitKey(1) == ord('q'):
            break

        # Capture frame-by-frame
        ret, frame = cap.read()

        threshold = cv2.getTrackbarPos("Threshold", "Window")
        print(threshold)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        show_binary_images(frame, threshold)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
