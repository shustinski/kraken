import os
from pathlib import Path

import PIL.Image
from PIL import Image
import numpy as np
import random

from lib.data_interfaces import CutSettings, SampleGenerationSettings, SamplePrepareSettings
from lib.file_func import filter_files


class SampleWorker:

    def __init__(self, path=None, paramns: CutSettings | None = None):
        self._path: str | None = path
        self._img_paths: list | None = None
        self._img_sizes: list[tuple[int, int]] | None = None
        self._params: CutSettings | None = paramns
        self._total_parts: int = 0

    def set_path(self, path:Path):
        if self._path == path:
            return
        if not path.is_dir():
            return
        self._path = path
        self._calculate_samples_from_path()

    def set_settings(self, setting: CutSettings):
        if self._params == setting:
            return
        self._params = setting
        self._calculate_len()

    def _calculate_len(self):
        if self._path is None:
            return
        if not os.path.isdir(self._path):
            return
        if self._params is None:
            return
        if self._img_sizes is None:
            self._calculate_samples_from_path()
        else:
            self.calculate_samples_amount()

    def __len__(self):
        self._calculate_len()
        return self._total_parts

    def _calculate_samples_from_path(self):
        """

        Args:
            path: parh for image calculation

        """

        valid_extensions = ('.jpg', '.png')

        self._img_paths = filter_files(self._path, valid_extensions)
        if len(self._img_paths) == 0:
            return

        self._get_image_sizes()

        self.calculate_samples_amount()

    def _get_image_sizes(self):
        self._img_sizes = []
        for image in self._img_paths:
            with Image.open(image) as img:
                height, width = img.size
            size = (height, width)
            self._img_sizes.append(size)

    def calculate_samples_amount(self):
        if self._params is None or self._path is None:
            return 0

        self._total_parts = 0
        for image_size in self._img_sizes:
            parts_in_one_image = self.calculate_image_parts(image_size)
            self._total_parts += parts_in_one_image

    def calculate_image_parts(self, image_size: tuple[int, int]) -> int:
        """

        Args:
            image_size: (height,width)

        Returns:
            number of parts
        """

        im_height, im_width = image_size

        step = self._params.step
        sample_x_size = self._params.x_size
        sample_y_size = self._params.y_size

        width_steps = int((im_width - sample_x_size) / step) + 1
        height_steps = int((im_height - sample_y_size) / step) + 1

        frames_in_frame = width_steps * height_steps

        vertical_frames = horizontal_frames = 0

        if self._params.vertical_rotation:
            vertical_frames = frames_in_frame

        if self._params.horizontal_rotation:
            horizontal_frames = 2 * frames_in_frame

        frames_in_frame += horizontal_frames + vertical_frames

        return frames_in_frame

class SampleCalculator:

    def __init__(self, image_size:tuple[int,int], params:SampleGenerationSettings):
        self._image_size = image_size
        self._params = params
        self._width_steps = 0
        self._height_steps = 0

    @classmethod
    def from_path(cls, path:str, params:SampleGenerationSettings):
        with Image.open(path) as image:
            size = image.size
        return cls(size, params)

    def __len__(self):
        return self._calculate_parts()

    @property
    def size(self):
        return self._width_steps, self._height_steps

    def _calculate_parts(self):
        im_height, im_width = self._image_size

        step = self._params.step
        sample_x_size, sample_y_size = self._params.segment_size

        self._width_steps = int((im_width - sample_x_size) / step) + 1
        self._height_steps = int((im_height - sample_y_size) / step) + 1

        frames_in_frame = self._width_steps * self._height_steps

        vertical_frames = horizontal_frames = 0

        if self._params.vertical_rotation:
            vertical_frames = frames_in_frame

        if self._params.horizontal_rotation:
            horizontal_frames = 2 * frames_in_frame

        frames_in_frame += horizontal_frames + vertical_frames

        return frames_in_frame



class ImagePreparator:

    def __init__(self, path:Path, parameters:SamplePrepareSettings):
        self._params = parameters
        self._path = path

    @property
    def image(self):
        self._prepare()
        return self._image

    @property
    def size(self):
        return self._lazy_size()

    def _lazy_size(self):
        if self._params.target_size is not None:
            return self._params.target_size
        with Image.open(self._path) as img:
            self._image = img
        if self._params.edge_cut is None:
            return self._image.size
        return self._size_after_crop()

    def _size_after_crop(self):
        return tuple(np.subtract(self._image.size,self._params.edge_cut))

    def _prepare(self):
        with Image.open(self._path) as img:
            img.load()
        self._image = img
        self._size = img.size
        if self._params.edge_cut is not None:
           self._crop()
        if self._params.target_size is not None:
            self._resize()

    def _crop(self):
        p = self._params
        top_left_x = p.edge_cut[0]
        top_left_y = p.edge_cut[1]
        bottom_right_x =  self._size[0] - p.edge_cut[0]
        bottom_right_y =  self._size[1] - p.edge_cut[1]
        self._image = self._image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))

    def _resize(self):
        p = self._params
        self._image = self._image.resize(p.target_size)


class SampleFastCutter:
    """
    Рассчитывет число образцов в кадре и возвращет необходимый кусочк по методу __getitem__
    Соотвутствующий кусочек рассчитывается исходя из парметров
    """

    def __init__(self, matrix:tuple[np.ndarray,np.ndarray], parameters: SampleGenerationSettings, shuffle=False):

        self._params = parameters
        self._shuffle = shuffle
        self.image_matrix = matrix[0]
        self.label_matrix = matrix[1]
        self.base_size = (self.image_matrix.shape[1],self.image_matrix.shape[2])

        self.sample = SampleCalculator(self.base_size,parameters)
        parts = len(self.sample)
        parts_list = list(range(parts))
        if shuffle:
            random.shuffle(parts_list)
        self._parts_list = parts_list

    @classmethod
    def from_image(cls, sample_pair:tuple[PIL.Image.Image,PIL.Image.Image], parameters: SampleGenerationSettings, shuffle=False):
        image_matrix = cls.get_matrix_from_image(sample_pair[0],parameters.channels)
        label_matrix = cls.get_matrix_from_image(sample_pair[1],parameters.channels)
        return cls((image_matrix,label_matrix),parameters, shuffle)

    def __getitem__(self, item):
        loc = self._parts_list[item]
        location, rotation_index = self._define_location_based_on_index(loc)
        width_steps, height_steps = self.sample.size
        sample_x, sample_y = self._params.segment_size

        row = location//width_steps
        col = location - width_steps*row

        left = col * self._params.step
        right = left + sample_x
        top = row * self._params.step
        bottom = top + sample_y

        if right >= self.base_size[0] and bottom >= self.base_size[1]:
            image = self.image_matrix[:, -sample_y:, -sample_x:].copy()
            label = self.label_matrix[:, -sample_y:, -sample_x:].copy()
        elif right > self.base_size[0]:
            image = self.image_matrix[:, top:bottom, -sample_x:].copy()
            label = self.label_matrix[:, top:bottom, -sample_x:].copy()
        elif bottom > self.base_size[1]:
            image = self.image_matrix[:, -sample_y:, left:right].copy()
            label = self.label_matrix[:, -sample_y:, left:right].copy()
        else:
            image = self.image_matrix[:, top:bottom, left:right].copy()
            label = self.label_matrix[:, top:bottom, left:right].copy()

        if rotation_index != 0:
            image[0] = np.rot90(image[0], k=rotation_index)
            label[0] = np.rot90(label[0], k=rotation_index)

        return image,label

    @staticmethod
    def get_matrix_from_image(img, channels):
        matrix = np.array(img).astype('float32')
        matrix = matrix/256
        if channels == 1:
            matrix = np.reshape(matrix, (channels, matrix.shape[0], matrix.shape[1]))
        else:
            matrix = matrix.transpose(2, 0, 1)

        return  matrix

    def __len__(self):
        return len(self._parts_list)

    def _define_location_based_on_index(self, part_number):
        """
        Args:
            part_number:

        Returns:возвращает положение выбронного участка
            в пространстве в зависмости от параметра вращения,
            а также угол поворота

        Индекс вращений соотвутствует числу поворотов на 90 градусов
        поэтому для обоих видов вращений просто берем остаток от деления на четыре
        для горизонтального вращения формула 2 - part_number % 3 дает возможные варианты (-1,0,1),
        что соотвутствует вращению на (-90,0,90)
        для вертикального вращения формула 2*(part_number % 2) дает возможные варианты (0,2),
        что соотвутствует вращению на (0,180)
        """
        if self._params.vertical_rotation and self._params.horizontal_rotation:
            location =  part_number // 4
            rotation_index = part_number % 4
        elif self._params.horizontal_rotation:
            location = part_number // 3
            rotation_index = 2 - part_number % 3
        elif self._params.vertical_rotation:
            location = part_number // 2
            rotation_index = 2*(part_number % 2)
        else:
            location = part_number
            rotation_index = 0

        return location,rotation_index


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
