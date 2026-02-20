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
    channels, segment_width, segment_height = segment_size
    base_height = base_image.shape[1]
    base_width = base_image.shape[2]
    stride_height = max(1, int(segment_height - overlap))
    stride_width = max(1, int(segment_width - overlap))

    row_steps = int(base_height / stride_height) + 1
    column_steps = int(base_width / stride_width) + 1

    fragments = row_steps * column_steps
    # Tensor layout is (N, C, H, W)
    images = np.zeros((fragments, channels, segment_height, segment_width), dtype=base_image.dtype)

    for row in range(row_steps):
        for col in range(column_steps):
            image_index = row * column_steps + col
            left = col * stride_width
            right = left + segment_width
            top = row * stride_height
            bottom = top + segment_height

            src_top = top if bottom <= base_height else max(0, base_height - segment_height)
            src_left = left if right <= base_width else max(0, base_width - segment_width)
            src_bottom = min(base_height, src_top + segment_height)
            src_right = min(base_width, src_left + segment_width)

            patch = np.zeros((channels, segment_height, segment_width), dtype=base_image.dtype)
            copy_height = max(0, int(src_bottom - src_top))
            copy_width = max(0, int(src_right - src_left))
            if copy_height > 0 and copy_width > 0:
                patch[:, :copy_height, :copy_width] = base_image[:, src_top:src_bottom, src_left:src_right]
            images[image_index] = patch

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
    base_width = int(base_image[0])
    base_height = int(base_image[1])
    result = np.zeros((base_height, base_width), dtype=np.float32)

    # Predictions layout is (N, C, H, W)
    segment_height = int(predictions.shape[2])
    segment_width = int(predictions.shape[3])
    stride_height = max(1, int(segment_height - overlap))
    stride_width = max(1, int(segment_width - overlap))
    row_steps = int(base_height / stride_height) + 1
    column_steps = int(base_width / stride_width) + 1

    # If overlap is too large we still keep at least one pixel in each direction.
    raw_crop = int(overlap / 2) if overlap % 2 == 0 else int(overlap / 2) + 1
    crop_height = min(raw_crop, max(0, segment_height // 2))
    crop_width = min(raw_crop, max(0, segment_width // 2))
    parts_count = int(predictions.shape[0])

    for row in range(row_steps):
        for col in range(column_steps):
            sewed_part_index = row * column_steps + col
            if sewed_part_index >= parts_count:
                continue

            left = col * stride_width
            right = left + segment_width
            top = row * stride_height
            bottom = top + segment_height

            src_top = top if bottom <= base_height else max(0, base_height - segment_height)
            src_left = left if right <= base_width else max(0, base_width - segment_width)

            top_crop = 0 if row == 0 else crop_height
            bottom_crop = 0 if row == (row_steps - 1) else crop_height
            left_crop = 0 if col == 0 else crop_width
            right_crop = 0 if col == (column_steps - 1) else crop_width

            dst_top = src_top + top_crop
            dst_bottom = src_top + segment_height - bottom_crop
            dst_left = src_left + left_crop
            dst_right = src_left + segment_width - right_crop

            dst_top = max(0, min(base_height, dst_top))
            dst_bottom = max(0, min(base_height, dst_bottom))
            dst_left = max(0, min(base_width, dst_left))
            dst_right = max(0, min(base_width, dst_right))

            if dst_bottom <= dst_top or dst_right <= dst_left:
                continue

            src_patch_top = dst_top - src_top
            src_patch_left = dst_left - src_left
            src_patch_bottom = src_patch_top + (dst_bottom - dst_top)
            src_patch_right = src_patch_left + (dst_right - dst_left)

            patch = predictions[sewed_part_index, 0, :, :]
            result[dst_top:dst_bottom, dst_left:dst_right] = patch[
                src_patch_top:src_patch_bottom,
                src_patch_left:src_patch_right,
            ]
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    result = np.clip(result, 0.0, 1.0)
    result = result * 255
    # result = result.reshape(base_height, base_width)
    resimg = Image.fromarray(result.astype('uint8'), mode='L')
    return resimg
