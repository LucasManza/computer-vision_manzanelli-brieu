import numpy as np
import cv2 as cv


def __resize_img__(frame, bbox, width, height):
    img_cropped = frame[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    img_cropped = cv.resize(img_cropped, (width, height))
    return img_cropped


def __concat_images__(images: list, width, height, size):
    acc_hor_imgs = []
    while len(images) > 0:
        imgs_aux = images[:size]

        if imgs_aux.__len__() < size:
            for i in range(0, (5 - imgs_aux.__len__())):
                imgs_aux.append(np.zeros((width, height, 3), np.uint8))

        horizontal_concat = np.concatenate(imgs_aux, axis=1)
        acc_hor_imgs.append(horizontal_concat)
        del images[:size]

    mosaic = np.concatenate(acc_hor_imgs, axis=0)
    return mosaic


def create_mosaic(dictionary, img_size, mosaic_size):
    imgs = map(lambda d: __resize_img__(d["frame"], d["bbox"], img_size, img_size), dictionary)
    imgs = list(imgs)

    mosaic = __concat_images__(imgs, img_size, img_size, mosaic_size)
    cv.imshow('Mosaic', mosaic)


if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    width = 200
    height = 200

    while True:
        if cv.waitKey(1) == ord('q'):
            break

        ret, frame = cap.read()
        img = __resize_img__(frame, [0, 100, 0, 100], width, height)
        img_2 = __resize_img__(frame, [100, 200, 0, 100], width, height)
        img_3 = __resize_img__(frame, [300, 400, 200, 300], width, height)
        img_4 = __resize_img__(frame, [100, 200, 0, 100], width, height)

        mosaic = __concat_images__([img, img_2, img_3, img_4], width, height, 2)
        cv.imshow('Mosaic', mosaic)
