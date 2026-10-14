"""images: save color."""
from __future__ import annotations
from PIL import Image
import numpy as np

def save_color(channel, color, path, image_type='JPEG', quality=95):
    """
    channel – (H, W) в диапазоне [0,1]
    color   – строка 'R', 'G' или 'B' – в какой цветовой канал записать данные
    """
    H, W = channel.shape
    # Создаём пустой (H, W, 3) массив из нулей (чёрный фон)
    img = np.zeros((H, W, 3), dtype=np.float32)

    # Заполняем нужный канал
    idx = {'R': 0, 'G': 1, 'B': 2}[color.upper()]
    img[..., idx] = channel

    # Приводим к uint8 и сохраняем
    img_uint8 = (img * 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode='RGB').save(path, format=image_type, quality=quality)

def save_binary(matrix, path, image_type='JPEG', quality=95):
    """
    channel – (H, W) в диапазоне [0,1]
    color   – строка 'R', 'G' или 'B' – в какой цветовой канал записать данные
    """

    # Приводим к uint8 и сохраняем
    img_uint8 = (matrix[0] * 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode='L').save(path, format=image_type, quality=quality)
