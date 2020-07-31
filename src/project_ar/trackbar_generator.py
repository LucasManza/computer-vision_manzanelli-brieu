import cv2


class TrackbarSettings:
    def __init__(self, window_name, value_name: str = "Threshold", value=127, max_value=255):
        self.window_name: str = window_name
        self.value_name = value_name
        self.value = value
        self.max_value = max_value
        self.__generate_trackbars__()

    def __nothing__(self, x):
        pass

    def __generate_trackbars__(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar(self.value_name, self.window_name, self.value, self.max_value, self.__nothing__)

    def update(self):
        self.value = cv2.getTrackbarPos(self.value_name, self.window_name)
        return self.value