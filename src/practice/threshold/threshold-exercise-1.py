# todo  1) Generar una imagen binaria normal y
#  otra invertida sobre la cámara, controlando el umbral con una barra deslizante


import cv2


# Show normal, binary and binary inverted images.
def show_binary_images(image):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
    ret, binary_inv = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow('original', gray_frame)
    cv2.imshow('binary', binary)
    cv2.imshow('binary_inv', binary_inv)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while True:

        if cv2.waitKey(1) == ord('q'):
            break

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('i'):
            show_binary_images(frame)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
