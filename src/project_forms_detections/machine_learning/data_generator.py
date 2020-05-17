import imutils

import project_forms_detections.image_operators.threshold_operators as threshold_operators


def __rotate_img__(img_target, quantity=3):
    result = []
    angle = 45
    for i in range(1, quantity):
        result.append(imutils.rotate(img_target, angle))
        angle = angle + 45
    return result


def generate_bin_img_training(img_target, others_img):
    bin_target = threshold_operators.generate_binary_image(img_target, 127)
    target_data = __rotate_img__(bin_target)
    others = list(map(lambda i: threshold_operators.generate_binary_image(i, 127)
                      , others_img))

    return target_data + others
