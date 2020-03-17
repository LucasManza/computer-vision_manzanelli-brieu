# todo  2) Aplicar los métodos de umbral automático


import cv2


# Show normal, binary and binary inverted images. With automatic OTSU thresh adjust.
def show_binary_images(image):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, binary_inv = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow('original + otsu', gray_frame)
    cv2.imshow('binary + otsu', binary)
    cv2.imshow('binary_inv + otsu', binary_inv)


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
