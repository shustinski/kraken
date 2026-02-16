import os

import numpy as np
import numpy.typing as npt
from PIL import Image


# Считывание файла с именами изображений
def get_names_from_file(path):
    ##    with open(path) as file:
    ##        img_names = [row.strip() for row in file]
    ##    return img_names

    f = open(path, 'r')
    for line in f:
        filename = line.rstrip('\n')
        yield filename
    f.close()


# Загрузка изображения
def get_imges(fname, dirname, img_rows, img_cols, ext):
    img_names = get_names_from_file(fname)
    imgs = []
    for img_name in img_names:
        path = os.path.join(dirname, img_name + ext)
        im = Image.open(path)  # Загрузка изображения
        if (im.size[0] != img_cols or im.size[1] != img_rows):  # Изменение размера, если неообходимо
            im.thumbnail([img_cols, img_rows])
        imgs.append(np.array(im).astype('float32'))  # Перобразование в формат np.array
    imgs = np.true_divide(imgs, 255)  # Нормализация
    return imgs


# Изменение числа каналов
def reshape_imgs(imgs):
    imgs = imgs.reshape((imgs.shape[0], imgs.shape[1], imgs.shape[2], -1, 1))
    # Значение "-1" означает, что по этому напрвлению размерность итоговой матрицы будет расчитываться исходя из исходной
    # в данном случае это позволяет одинаково обрабатывать как многоканальные RGB, так и одноканаольные L изображения
    imgs = imgs[:, :, :, 0]
    return imgs


# Разрезать большое входное изображение на массив маленьких входных картинок для НС
def cut_image(base_image, segment_size, overlap):
    # segment_size(width,height,channels)
    channels, segment_width, segment_height  = segment_size
    base_height = base_image.shape[1]
    base_width = base_image.shape[2]

    row_steps = int(base_height / (segment_height - overlap)) + 1
    column_steps = int(base_width / (segment_width - overlap)) + 1

    fragments = row_steps * column_steps
    images = np.zeros((fragments, channels, segment_width, segment_height))

    for row in range(row_steps):
        for col in range(column_steps):
            image_index = row * column_steps + col
            left = col * (segment_width - overlap)
            right = left + segment_width
            top = row * (segment_height - overlap)
            bottom = top + segment_height

            if right >= base_width and bottom >= base_height:
                images[image_index] = base_image[:,-segment_height:, -segment_width:]
            elif right > base_width:
                images[image_index] = base_image[:,top:bottom, -segment_width:]
            elif bottom > base_height:
                images[image_index] = base_image[:,-segment_height:, left:right]
            else:
                images[image_index] = base_image[:,top:bottom, left:right]

    return images / 255


# Срезать рамку с одноканального изображения (залить чёрным)
def img_crop_border(cropBorder, img):
    im_width = img.size[0]
    im_height = img.size[1]
    imgPix = img.load()  # Выгружаются значения пикселей

    result = np.zeros((im_height, im_width))

    for i in range(cropBorder, im_width - cropBorder, 1):
        for j in range(cropBorder, im_height - cropBorder, 1):
            result[j, i] = imgPix[i, j]

    img = Image.fromarray(result.astype('uint8'), mode=img.mode)
    return img


# base_image, segment_size, overlap
def sew_image(base_image, predictions: npt.ArrayLike, overlap) -> Image:
    """

    :param base_image:  (width,height)
    :param predictions:
    :param overlap:
    :return: sewed image
    """
    base_width = base_image[0]
    base_height = base_image[1]
    result = np.zeros((base_height, base_width))

    segment_width = predictions.shape[2]
    segment_height = predictions.shape[3]

    row_steps = int(base_height / (segment_width - overlap)) + 1
    column_steps = int(base_width / (segment_height - overlap)) + 1
    crop_border = int(overlap / 2) if overlap % 2 == 0 else int(overlap / 2) + 1

    for row in range(row_steps):
        for col in range(column_steps):
            # big image coordinates
            if row == 0:
                top_coord_big_img = crop_border
                bot_coord_big_img = segment_height - crop_border
            elif row == (row_steps - 1):
                top_coord_big_img = base_height - segment_height + crop_border
                bot_coord_big_img = base_height - crop_border
            else:
                top_coord_big_img = row * (segment_height - overlap) + crop_border
                bot_coord_big_img = row * (segment_height - overlap) + segment_height - crop_border

            if col == 0:
                left_coord_big_img = crop_border
                right_coord_big_img = segment_width - crop_border
            elif col == (column_steps - 1):
                left_coord_big_img = base_width - segment_width + crop_border
                right_coord_big_img = base_width - crop_border
            else:
                left_coord_big_img = col * (segment_width - overlap) + crop_border
                right_coord_big_img = col * (segment_width - overlap) + segment_width - crop_border

            sewed_part_index = row * column_steps + col
            if crop_border == 0:
                patch = predictions[sewed_part_index, 0, :, :]
            else:
                patch = predictions[
                    sewed_part_index,
                    0,
                    crop_border:-crop_border,
                    crop_border:-crop_border,
                ]
            result[top_coord_big_img:bot_coord_big_img, left_coord_big_img:right_coord_big_img] = patch
    result = result * 255
    # result = result.reshape(base_height, base_width)
    resimg = Image.fromarray(result.astype('uint8'), mode='L')
    return resimg
