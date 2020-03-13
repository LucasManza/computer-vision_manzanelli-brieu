import numpy as np
import cv2
import datetime

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('w'):
        time = datetime.datetime.now()
        cv2.imwrite('../assets/threshold-img.png', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
