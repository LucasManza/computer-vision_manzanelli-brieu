import enum
import cv2
from matplotlib import pyplot as plt


# 1)Al pulsar una tecla, obtener la plantilla recortando un cuadrado de 100 x 100 píxeles
# en el centro de la imagen y mostrarla en una ventana aparte
# 2) Aplicar matchTemplate con la plantilla sobre la imagen de la cámara y mostrar la imagen generada.
# Usar teclas para cambiar el método.
# 3)  Buscar la posición del macheo con minMaxLoc() y
# dibujar un recuadro de 100 x 100 sobre la imagen original, señalando la detección
# 4) Repetir usando dos plantillas sobre la cámara, recuadrando la detección con diferente color

def cut_roi_img(frame, size: int):
    height = frame.shape[0]
    width = frame.shape[1]
    min_height: int = int(height / 2 - size)
    max_height: int = int(height / 2 + size)
    min_width: int = int(width / 2 - size)
    max_width: int = int(width / 2 + size)
    roi = frame[min_height:max_height, min_width:max_width]
    cv2.imshow('', roi)
    return roi


red_color = (0, 0, 255)
blue_color = (255, 0, 0)


class TemplateMethods:
    SQ_DIFF = 'cv2.TM_SQDIFF_NORMED'
    CORR = 'cv2.TM_CCORR_NORMED'


def matching_template(image, template, method: str):
    res = cv2.matchTemplate(image, template, eval(method))
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    color = red_color
    top_left = min_loc

    if method == TemplateMethods.CORR:
        color = blue_color
        top_left = max_loc

    bottom_right = (top_left[0] + rectangle_size, top_left[1] + rectangle_size)
    return top_left, bottom_right, color


# EXECUTION

current_method: str = TemplateMethods.SQ_DIFF
roi = None
frame = None
rectangle_size: int = 100

cap = cv2.VideoCapture(0)

while True:

    if cv2.waitKey(1) == ord('q'):
        break

    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    if cv2.waitKey(1) == ord('1'):
        roi = cut_roi_img(frame, rectangle_size)
        current_method = TemplateMethods.SQ_DIFF

    if cv2.waitKey(1) == ord('2'):
        roi = cut_roi_img(frame, rectangle_size)
        current_method = TemplateMethods.CORR

    if roi is not None:
        matching_tuple = matching_template(roi, frame, current_method)
        frame = cv2.rectangle(frame, matching_tuple[0], matching_tuple[1], matching_tuple[2], 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
