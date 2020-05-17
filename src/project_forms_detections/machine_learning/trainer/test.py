import cv2
import numpy as np

from project_forms_detections.image_operators import hu_operators, threshold_operators
from project_forms_detections.machine_learning.data_generator import generate_bin_img_training
from project_forms_detections.machine_learning.trainer.svm_trainer import SVMTrainer


def __convert_UMat__(hu_moment):
    return cv2.UMat(np.array(hu_moment))


if __name__ == '__main__':
    trainer = SVMTrainer()

    labels = np.array([1, 1, -1, -1])
    # trainingData = np.matrix([[501, 10], [255, 10], [501, 255], [10, 501]], dtype=np.float32)

    star_img = cv2.imread('../../../assets/star.png')
    piece_img = cv2.imread('../../../assets/piece.png')
    background_img = cv2.imread('../../../assets/white_bacground.png')

    train_bin_imgs = generate_bin_img_training(star_img, [piece_img, background_img])
    # umat_all_hu = list(map(lambda i: __convert_UMat__(hu_operators.hu_moments(i)), train_bin_imgs))

    trainer.train(train_bin_imgs, labels)

    bin_start = threshold_operators.generate_binary_image(star_img, 127)

    trainer.predict(bin_start)
