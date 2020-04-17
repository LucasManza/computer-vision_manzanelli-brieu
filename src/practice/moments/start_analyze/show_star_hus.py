import practice.moments.start_analyze.hu_img_analyze as hu_analyze
import cv2

star_src: str = '../../../assets/star-image.png'
star_dataset: [str] = [
    '../../../assets/dataset/star/1.png',
    '../../../assets/dataset/star/2.png',
    '../../../assets/dataset/star/3.png',
    '../../../assets/dataset/star/4.png',
    '../../../assets/dataset/star/5.png',
    '../../../assets/dataset/star/6.png',
    '../../../assets/dataset/star/7.png',
    '../../../assets/dataset/star/8.png',
    '../../../assets/dataset/star/9.png',
]


def __next_img__(index: int) -> str:
    if index > 9:
        return star_src
    else:
        return star_dataset[index - 1]


def __nothing__(x):
    pass


if __name__ == '__main__':
    invert_bin_img: bool = False
    threshold = 127
    dilation_struct_size = 1

    frame = cv2.imread(star_src)

    dataset = [frame]
    for img_src in star_dataset:
        f = cv2.imread(img_src)
        dataset.append(f)

    hu_analyze.compare_images_hu(dataset, '', invert_bin_img, threshold, dilation_struct_size)
