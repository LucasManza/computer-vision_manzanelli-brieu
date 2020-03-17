# todo Exercise
#  1) Generar una imagen binaria normal y otra invertida sobre la cámara, controlando el umbral con una barra deslizante
#  2) Aplicar los métodos de umbral automático
#  3) Generar una imagen binaria normal sobre la cámara, usando ambos métodos de umbral adaptativo
#  , controlando el tamaño de bloque con una barra deslizante
#  4) Generar una imagen binaria con dos umbrales con inRange,
#  para segmentar un objeto por su color en el espacio HSV
#  5) En un ambiente controlado, controlando color de fondo e iluminación,
#  obtener una imagen binaria perfecta de piezas, con el método más adecuado

import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while True:

        if cv2.waitKey(1) == ord('q'):
            break

        # Capture frame-by-frame
        ret, frame = cap.read()

        # frame = cv2.rectangle(frame, matching_tuple[0], matching_tuple[1], matching_tuple[2], 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def generate_binary_images(image):
    thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
    thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
    thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)

    return [thresh1, thresh2, thresh3, thresh4, thresh5]
