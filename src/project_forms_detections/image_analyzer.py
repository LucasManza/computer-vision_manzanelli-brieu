import cv2

from project_forms_detections.image_operators import contours_operators as contours_operators
from project_forms_detections.image_operators import morphological_operators as morph_operators
from project_forms_detections.image_operators import threshold_operators as threshold_operators


class ImageAnalyzer:
    def __init__(self, window_name):
        self.threshold = 127
        self.morph_struct_size = 10
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

    def analyze_image(self, image, invert_image=False):
        # Generate monochromatic image
        bin_img = threshold_operators.generate_binary_image(image, self.threshold, invert_image)

        # Clean monochromatic by reduce noise using dilation and closing morph operators
        bin_clean_img = morph_operators.reduce_noise_dil_closing(bin_img, self.morph_struct_size)

        # Find contours over a binary image (clean)
        contours = contours_operators.find_contours(bin_clean_img)

        return bin_clean_img, contours