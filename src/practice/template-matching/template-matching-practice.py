import numpy as np
import cv2
import datetime


# 1)Al pulsar una tecla, obtener la plantilla recortando un cuadrado de 100 x 100 píxeles
# en el centro de la imagen y mostrarla en una ventana aparte
# 2) Aplicar matchTemplate con la plantilla sobre la imagen de la cámara y mostrar la imagen generada.
# Usar teclas para cambiar el método.
# 3)  Buscar la posición del macheo con minMaxLoc() y
# dibujar un recuadro de 100 x 100 sobre la imagen original, señalando la detección
# 4) Repetir usando dos plantillas sobre la cámara, recuadrando la detección con diferente color

def CutROIImg(frame, size: int):
    (height, width) = frame.shape[::-1]
    min_height: int = int(height / 2 - size)
    max_height: int = int(height / 2 + size)
    min_width: int = int(width / 2 - size)
    max_width: int = int(width / 2 + size)
    roi = frame[min_height:max_height, min_width:max_width]
    cv2.imshow('', roi)
    return roi


cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('c'):
        CutROIImg(frame, 100)

    if cv2.waitKey(1) == ord('w'):
        time = datetime.datetime.now()
        cv2.imwrite('../assets/threshold-img.png', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

