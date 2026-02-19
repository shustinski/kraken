import os
import importlib
from pathlib import Path

import PIL.Image
from PIL import Image
import numpy as np
import random

from lib.data_interfaces import CutSettings, SampleGenerationSettings, SamplePrepareSettings
from lib.file_func import filter_files

def _load_sample_fast_cutter_getitem():
    try:
        module = importlib.import_module('lib.sample_fast_cutter_pyx')
        getitem_fn = getattr(module, 'sample_fast_cutter_getitem', None)
        module_file = str(getattr(module, '__file__', '')).lower()
        is_accelerated = module_file.endswith('.pyd') or module_file.endswith('.so')
        return getitem_fn, is_accelerated
    except Exception:
        return None, False


_sample_fast_cutter_getitem, _sample_fast_cutter_accelerated = _load_sample_fast_cutter_getitem()


def is_sample_fast_cutter_accelerated() -> bool:
    return bool(_sample_fast_cutter_accelerated and _sample_fast_cutter_getitem is not None)


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
        if getattr(self._params, 'additional_augmentation', False):
            frames_in_frame *= 2

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
        if self._params.additional_augmentation:
            frames_in_frame *= 2

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
        if self._params.enable_resize and self._params.target_size is not None:
            return self._params.target_size
        with Image.open(self._path) as img:
            size = img.size
        if not self._params.enable_crop or self._params.edge_cut is None:
            return size
        return self._size_after_crop(size)

    def _size_after_crop(self, image_size: tuple[int, int]):
        return tuple(np.subtract(image_size, self._params.edge_cut))

    def _prepare(self):
        with Image.open(self._path) as img:
            img.load()
            self._image = img.copy()
        self._size = img.size
        if self._params.enable_crop and self._params.edge_cut is not None:
           self._crop()
        if self._params.enable_resize and self._params.target_size is not None:
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
        # image_matrix shape: (C, H, W)
        self.base_size = (self.image_matrix.shape[1], self.image_matrix.shape[2])
        self._base_h = self.image_matrix.shape[1]
        self._base_w = self.image_matrix.shape[2]
        self._step = int(parameters.step)
        self._sample_x = int(parameters.segment_size[0])
        self._sample_y = int(parameters.segment_size[1])
        self._vertical_rotation = bool(parameters.vertical_rotation)
        self._horizontal_rotation = bool(parameters.horizontal_rotation)
        self._additional_augmentation = bool(getattr(parameters, 'additional_augmentation', False))
        self._augmentation_brightness_strength = max(
            0.0, float(getattr(parameters, 'augmentation_brightness_strength', 0.1))
        )
        self._augmentation_contrast_strength = max(
            0.0, float(getattr(parameters, 'augmentation_contrast_strength', 0.1))
        )
        self._augmentation_noise_probability = float(getattr(parameters, 'augmentation_noise_probability', 0.5))
        self._augmentation_noise_probability = min(1.0, max(0.0, self._augmentation_noise_probability))
        self._augmentation_noise_sigma = max(0.0, float(getattr(parameters, 'augmentation_noise_sigma', 0.01)))

        self.sample = SampleCalculator(self.base_size,parameters)
        parts = len(self.sample)
        self._width_steps, self._height_steps = self.sample.size
        parts_list = list(range(parts))
        if shuffle:
            random.shuffle(parts_list)
        self._parts_list = parts_list
        # Cython accelerator does not account for additional augmentation indexing.
        self._use_accelerator = is_sample_fast_cutter_accelerated() and not self._additional_augmentation

    @classmethod
    def from_image(cls, sample_pair:tuple[PIL.Image.Image,PIL.Image.Image], parameters: SampleGenerationSettings, shuffle=False):
        image_matrix = cls.get_matrix_from_image(sample_pair[0],parameters.channels)
        label_matrix = cls.get_matrix_from_image(sample_pair[1],parameters.channels)
        return cls((image_matrix,label_matrix),parameters, shuffle)

    def __getitem__(self, item):
        if self._use_accelerator and _sample_fast_cutter_getitem is not None:
            try:
                image, label = _sample_fast_cutter_getitem(
                    self._parts_list,
                    item,
                    self._vertical_rotation,
                    self._horizontal_rotation,
                    self._width_steps,
                    self._step,
                    self._sample_x,
                    self._sample_y,
                    self._base_w,
                    self._base_h,
                    self.image_matrix,
                    self.label_matrix,
                )
                if self._additional_augmentation:
                    image = self._apply_additional_augmentation(image)
                return image, label
            except Exception:
                # Disable accelerator for this instance after first failure.
                self._use_accelerator = False

        loc = self._parts_list[item]
        augmentation_variant = 0
        if self._additional_augmentation:
            loc, augmentation_variant = divmod(loc, 2)

        if self._vertical_rotation and self._horizontal_rotation:
            location = loc // 4
            rotation_index = loc % 4
        elif self._horizontal_rotation:
            location = loc // 3
            rotation_index = 2 - (loc % 3)
        elif self._vertical_rotation:
            location = loc // 2
            rotation_index = 2 * (loc % 2)
        else:
            location = loc
            rotation_index = 0

        row, col = divmod(location, self._width_steps)
        left = col * self._step
        top = row * self._step
        right = left + self._sample_x
        bottom = top + self._sample_y

        # Fast path: no border correction needed.
        if right <= self._base_w and bottom <= self._base_h:
            image = self.image_matrix[:, top:bottom, left:right].copy()
            label = self.label_matrix[:, top:bottom, left:right].copy()
        elif right > self._base_w and bottom > self._base_h:
            image = self.image_matrix[:, -self._sample_y:, -self._sample_x:].copy()
            label = self.label_matrix[:, -self._sample_y:, -self._sample_x:].copy()
        elif right > self._base_w:
            image = self.image_matrix[:, top:bottom, -self._sample_x:].copy()
            label = self.label_matrix[:, top:bottom, -self._sample_x:].copy()
        else:
            image = self.image_matrix[:, -self._sample_y:, left:right].copy()
            label = self.label_matrix[:, -self._sample_y:, left:right].copy()

        if rotation_index != 0:
            if rotation_index == 1:
                image[0] = np.flipud(image[0].T)
                label[0] = np.flipud(label[0].T)
            elif rotation_index == 2:
                image[0] = image[0, ::-1, ::-1]
                label[0] = label[0, ::-1, ::-1]
            else:
                image[0] = np.fliplr(image[0].T)
                label[0] = np.fliplr(label[0].T)

        if self._additional_augmentation and augmentation_variant == 1:
            image = self._apply_additional_augmentation(image)

        return image,label

    def _apply_additional_augmentation(self, image: np.ndarray) -> np.ndarray:
        # Lightweight photometric augmentation; labels remain unchanged.
        img = image.copy()
        brightness = random.uniform(
            1.0 - self._augmentation_brightness_strength,
            1.0 + self._augmentation_brightness_strength,
        )
        contrast = random.uniform(
            1.0 - self._augmentation_contrast_strength,
            1.0 + self._augmentation_contrast_strength,
        )
        img *= brightness
        mean = img.mean(axis=(1, 2), keepdims=True)
        img = (img - mean) * contrast + mean
        if random.random() < self._augmentation_noise_probability and self._augmentation_noise_sigma > 0.0:
            img += np.random.normal(0.0, self._augmentation_noise_sigma, size=img.shape).astype(np.float32)
        np.clip(img, 0.0, 1.0, out=img)
        return img.astype(np.float32, copy=False)

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
