class RGBColour:

    def __init__(self, r, g, b):
        red_value = self.__apply_value__(r)
        blue_value = self.__apply_value__(b)
        green_value = self.__apply_value__(g)

        self.value = (blue_value, green_value, red_value)

    def __apply_value__(self, value):
        if value > 255:
            return 255
        elif value < 0:
            return 0
        else:
            return value
