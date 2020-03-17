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


class MatchTemplateMethod(enum.Enum):
    CORR = 'TM_CCORR_NORMED',
    DIFF_SQR = 'TM_SQDIFF_NORMED'


# EXECUTION

current_method: str = 'cv2.TM_SQDIFF_NORMED'
roi = None
frame = None
blue_color = (255, 0, 0)
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

    if roi is not None:
        res = cv2.matchTemplate(roi, frame, eval(current_method))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = min_loc
        bottom_right = (top_left[0] + rectangle_size, top_left[1] + rectangle_size)

        frame = cv2.rectangle(frame, top_left, bottom_right, blue_color, 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
