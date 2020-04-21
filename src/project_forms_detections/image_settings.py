import cv2


class ImageSettings:
    def __init__(self, window_name, threshold=127, morph_struct_size=1):
        self.threshold = threshold
        self.morph_struct_size = morph_struct_size if morph_struct_size > 1 else 1
        self.approx_poly_dp = 1
        self.window_name: str = window_name
        self.__generate_trackbars__()

    def __nothing__(self, x):
        pass

    def __generate_trackbars__(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Threshold", self.window_name, self.threshold, 255, self.__nothing__)
        cv2.createTrackbar("Structure Size", self.window_name, self.morph_struct_size, 50, self.__nothing__)
        cv2.createTrackbar("AproxPolyDP", self.window_name, self.approx_poly_dp, 50, self.__nothing__)

    def __update_values__(self):
        self.threshold = cv2.getTrackbarPos("Threshold", self.window_name)
        self.morph_struct_size = cv2.getTrackbarPos("Structure Size", self.window_name)
        self.morph_struct_size = 1 if self.morph_struct_size < 2 else self.morph_struct_size

    def update(self, image):
        self.__update_values__()
        cv2.imshow(self.window_name, image)
