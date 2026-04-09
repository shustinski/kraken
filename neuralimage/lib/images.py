import os
import importlib
from pathlib import Path

import PIL.Image
from PIL import Image, ImageFilter
import numpy as np
import random

from lib.data_interfaces import CutSettings, SampleGenerationSettings, SamplePrepareSettings
from lib.file_func import filter_files
from lib.file_retry import retry_file_read


def _build_enabled_transform_variants(
    *,
    enable_rotate_180: bool,
    enable_rotate_90: bool,
    enable_flip_x: bool,
    enable_flip_y: bool,
    square_patch: bool,
) -> tuple[str, ...]:
    variants: list[str] = ['identity']
    if bool(enable_rotate_180):
        variants.append('rotate_180')
    if bool(enable_rotate_90) and bool(square_patch):
        variants.extend(('rotate_90', 'rotate_270'))
    if bool(enable_flip_x):
        variants.append('flip_x')
    if bool(enable_flip_y):
        variants.append('flip_y')
    return tuple(variants)


def _apply_transform_variant(patch: np.ndarray, variant: str) -> np.ndarray:
    if variant == 'rotate_180':
        return patch[:, ::-1, ::-1].copy()
    if variant == 'rotate_90':
        return np.rot90(patch, k=1, axes=(1, 2)).copy()
    if variant == 'rotate_270':
        return np.rot90(patch, k=-1, axes=(1, 2)).copy()
    if variant == 'flip_x':
        return patch[:, ::-1, :].copy()
    if variant == 'flip_y':
        return patch[:, :, ::-1].copy()
    return patch

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


def _resolve_crops_per_image(params: object) -> int:
    return max(1, int(getattr(params, 'crops_per_image', 64)))


class SampleWorker:
    VALID_EXTENSIONS = ('.jpg', '.png')

    def __init__(self, path=None, paramns: CutSettings | None = None):
        self._path: Path | None = Path(path) if path is not None else None
        self._img_paths: list[Path] | None = None
        self._img_sizes: list[tuple[int, int]] | None = None
        self._params: CutSettings | None = paramns
        self._total_parts: int = 0

    def set_path(self, path: Path | None):
        normalized_path = Path(path) if path is not None else None
        if self._path == normalized_path:
            return
        self._path = normalized_path
        self._img_paths = None
        self._img_sizes = None
        self._total_parts = 0

    def set_settings(self, setting: CutSettings):
        if self._params == setting:
            return
        self._params = setting
        self._total_parts = 0

    def _calculate_len(self):
        if self._path is None:
            self._total_parts = 0
            return
        if not os.path.isdir(self._path):
            self._total_parts = 0
            return
        if self._params is None:
            self._total_parts = 0
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

        self._img_paths = self.collect_image_paths(self._path)
        if len(self._img_paths) == 0:
            self._img_sizes = []
            self._total_parts = 0
            return

        self._img_sizes = self.collect_image_sizes(self._img_paths)
        self.calculate_samples_amount()

    @classmethod
    def collect_image_paths(cls, path: Path) -> list[Path]:
        if not path.is_dir():
            return []
        return filter_files(path, cls.VALID_EXTENSIONS)

    @staticmethod
    def collect_image_sizes(image_paths: list[Path]) -> list[tuple[int, int]]:
        image_sizes: list[tuple[int, int]] = []
        for image in image_paths:
            with retry_file_read(lambda: Image.open(image), path=image) as img:
                width, height = img.size
            image_sizes.append((height, width))
        return image_sizes

    def _get_image_sizes(self):
        self._img_sizes = self.collect_image_sizes(self._img_paths or [])

    def calculate_samples_amount(self):
        if self._params is None or self._path is None:
            self._total_parts = 0
            return 0

        self._total_parts = self.calculate_total_samples(self._img_sizes or [], self._params)
        return self._total_parts

    @classmethod
    def calculate_total_samples(
        cls,
        image_sizes: list[tuple[int, int]],
        params: CutSettings | None,
    ) -> int:
        if params is None:
            return 0

        total_parts = 0
        for image_size in image_sizes:
            total_parts += cls.calculate_image_parts_for_settings(image_size, params)
        return total_parts

    def calculate_image_parts(self, image_size: tuple[int, int]) -> int:
        return self.calculate_image_parts_for_settings(image_size, self._params)

    @staticmethod
    def calculate_image_parts_for_settings(
        image_size: tuple[int, int],
        params: CutSettings | None,
    ) -> int:
        """

        Args:
            image_size: (height,width)

        Returns:
            number of parts
        """

        im_height, im_width = image_size
        if params is None:
            return 0

        if getattr(params, 'random_crop', False):
            frames_in_frame = _resolve_crops_per_image(params)
        else:
            step = params.step
            sample_x_size = params.x_size
            sample_y_size = params.y_size

            width_steps = int((im_width - sample_x_size) / step) + 1
            height_steps = int((im_height - sample_y_size) / step) + 1
            frames_in_frame = width_steps * height_steps

        transform_variants = _build_enabled_transform_variants(
            enable_rotate_180=bool(getattr(params, 'vertical_rotation', False)),
            enable_rotate_90=bool(getattr(params, 'horizontal_rotation', False)),
            enable_flip_x=bool(getattr(params, 'flip_x', False)),
            enable_flip_y=bool(getattr(params, 'flip_y', False)),
            square_patch=int(params.x_size) == int(params.y_size),
        )
        frames_in_frame *= len(transform_variants)
        if getattr(params, 'scale_augmentation', False):
            frames_in_frame *= 2
        if getattr(params, 'additional_augmentation', False):
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
        with retry_file_read(lambda: Image.open(path), path=path) as image:
            width, height = image.size
        return cls((height, width), params)

    def __len__(self):
        return self._calculate_parts()

    @property
    def size(self):
        return self._width_steps, self._height_steps

    def _calculate_parts(self):
        if getattr(self._params, 'random_crop', False):
            self._width_steps = _resolve_crops_per_image(self._params)
            self._height_steps = 1
            frames_in_frame = self._width_steps
        else:
            im_height, im_width = self._image_size

            step = self._params.step
            sample_x_size, sample_y_size = self._params.segment_size

            self._width_steps = int((im_width - sample_x_size) / step) + 1
            self._height_steps = int((im_height - sample_y_size) / step) + 1
            frames_in_frame = self._width_steps * self._height_steps

        transform_variants = _build_enabled_transform_variants(
            enable_rotate_180=bool(getattr(self._params, 'vertical_rotation', False)),
            enable_rotate_90=bool(getattr(self._params, 'horizontal_rotation', False)),
            enable_flip_x=bool(getattr(self._params, 'flip_x', False)),
            enable_flip_y=bool(getattr(self._params, 'flip_y', False)),
            square_patch=tuple(getattr(self._params, 'segment_size', (0, 0)))[0]
            == tuple(getattr(self._params, 'segment_size', (0, 0)))[1],
        )
        frames_in_frame *= len(transform_variants)
        if self._params.scale_augmentation:
            frames_in_frame *= 2
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
        with retry_file_read(lambda: Image.open(self._path), path=self._path) as img:
            size = img.size
        if not self._params.enable_crop or self._params.edge_cut is None:
            return size
        return self._size_after_crop(size)

    def _size_after_crop(self, image_size: tuple[int, int]):
        crop_x, crop_y = int(self._params.edge_cut[0]), int(self._params.edge_cut[1])
        width = max(0, int(image_size[0]) - (crop_x * 2))
        height = max(0, int(image_size[1]) - (crop_y * 2))
        return width, height

    def _prepare(self):
        with retry_file_read(lambda: Image.open(self._path), path=self._path) as img:
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

    def __init__(
        self,
        matrix:tuple[np.ndarray,np.ndarray],
        parameters: SampleGenerationSettings,
        shuffle=False,
        skip_uniform_labels: bool = False,
        rare_mask_matrix: np.ndarray | None = None,
        rare_patch_oversampling_factor: int = 1,
    ):

        self._params = parameters
        self._shuffle = shuffle
        self.image_matrix = matrix[0]
        self.label_matrix = matrix[1]
        self.rare_mask_matrix = rare_mask_matrix
        # image_matrix shape: (C, H, W)
        self.base_size = (self.image_matrix.shape[1], self.image_matrix.shape[2])
        self._base_h = self.image_matrix.shape[1]
        self._base_w = self.image_matrix.shape[2]
        self._step = int(parameters.step)
        self._sample_x = int(parameters.segment_size[0])
        self._sample_y = int(parameters.segment_size[1])
        self._square_patch = self._sample_x == self._sample_y
        self._vertical_rotation = bool(parameters.vertical_rotation)
        self._horizontal_rotation = bool(parameters.horizontal_rotation)
        self._flip_x = bool(getattr(parameters, 'flip_x', False))
        self._flip_y = bool(getattr(parameters, 'flip_y', False))
        self._skip_uniform_labels = bool(skip_uniform_labels)
        self._rare_patch_oversampling_factor = max(1, int(rare_patch_oversampling_factor))
        self._random_crop = bool(getattr(parameters, 'random_crop', False))
        self._scale_augmentation = bool(getattr(parameters, 'scale_augmentation', False))
        self._scale_augmentation_strength = max(
            0.0, float(getattr(parameters, 'scale_augmentation_strength', 0.2))
        )
        self._additional_augmentation = bool(getattr(parameters, 'additional_augmentation', False))
        self._augmentation_brightness_strength = max(
            0.0, float(getattr(parameters, 'augmentation_brightness_strength', 0.1))
        )
        self._augmentation_contrast_strength = max(
            0.0, float(getattr(parameters, 'augmentation_contrast_strength', 0.1))
        )
        self._augmentation_gamma_strength = max(
            0.0, float(getattr(parameters, 'augmentation_gamma_strength', 0.15))
        )
        self._augmentation_noise_probability = float(getattr(parameters, 'augmentation_noise_probability', 0.5))
        self._augmentation_noise_probability = min(1.0, max(0.0, self._augmentation_noise_probability))
        self._augmentation_noise_sigma = max(0.0, float(getattr(parameters, 'augmentation_noise_sigma', 0.01)))
        self._augmentation_blur_probability = min(
            1.0,
            max(0.0, float(getattr(parameters, 'augmentation_blur_probability', 0.25))),
        )
        self._augmentation_blur_radius = max(
            0.0,
            float(getattr(parameters, 'augmentation_blur_radius', 1.0)),
        )

        self.sample = SampleCalculator(self.base_size,parameters)
        parts = len(self.sample)
        self._width_steps, self._height_steps = self.sample.size
        self._base_locations = max(0, self._width_steps * self._height_steps)
        self._transform_variants = _build_enabled_transform_variants(
            enable_rotate_180=self._vertical_rotation,
            enable_rotate_90=self._horizontal_rotation,
            enable_flip_x=self._flip_x,
            enable_flip_y=self._flip_y,
            square_patch=self._square_patch,
        )
        self._scale_variants = 2 if self._scale_augmentation else 1
        self._base_crop_specs = (
            self._build_base_crop_specs()
            if (self._random_crop or self._scale_augmentation) and self._base_locations > 0
            else None
        )
        self._scaled_crop_specs = (
            self._build_scaled_crop_specs()
            if self._scale_augmentation and self._base_locations > 0
            else None
        )
        parts_list = self._build_parts_list(parts)
        if shuffle:
            random.shuffle(parts_list)
        self._parts_list = parts_list
        # The current accelerator implementation rotates only channel 0 and can
        # fail on non-square 90/270 rotations.
        self._use_accelerator = (
            is_sample_fast_cutter_accelerated()
            and not self._additional_augmentation
            and self.image_matrix.shape[0] == 1
            and (self._square_patch or not self._horizontal_rotation)
        )

    @classmethod
    def from_image(
        cls,
        sample_pair:tuple[PIL.Image.Image,PIL.Image.Image],
        parameters: SampleGenerationSettings,
        shuffle=False,
        skip_uniform_labels: bool = False,
        rare_mask: PIL.Image.Image | None = None,
        rare_patch_oversampling_factor: int = 1,
    ):
        image_matrix = cls.get_matrix_from_image(sample_pair[0],parameters.channels)
        label_matrix = cls.get_matrix_from_image(sample_pair[1], 1)
        rare_mask_matrix = None
        if rare_mask is not None:
            rare_mask_matrix = cls.get_matrix_from_image(rare_mask, 1)
        return cls(
            (image_matrix, label_matrix),
            parameters,
            shuffle,
            skip_uniform_labels=skip_uniform_labels,
            rare_mask_matrix=rare_mask_matrix,
            rare_patch_oversampling_factor=rare_patch_oversampling_factor,
        )

    def _build_parts_list(self, parts: int) -> list[int]:
        needs_custom_parts = (
            self._skip_uniform_labels
            or (
                self.rare_mask_matrix is not None
                and self._rare_patch_oversampling_factor > 1
            )
        )
        if not needs_custom_parts or self._base_locations <= 0:
            return list(range(parts))

        parts_list: list[int] = []
        for location in range(self._base_locations):
            for variant_index in range(len(self._transform_variants)):
                base_variant = (location * len(self._transform_variants)) + variant_index
                for scale_variant in range(self._scale_variants):
                    if self._skip_uniform_labels and self._is_uniform_label_location(
                        location,
                        scale_variant=scale_variant,
                    ):
                        continue
                    scaled_variant = (base_variant * self._scale_variants) + scale_variant
                    repeat_count = 1
                    if self._is_rare_location(location, scale_variant=scale_variant):
                        repeat_count = self._rare_patch_oversampling_factor
                    for _ in range(repeat_count):
                        if self._additional_augmentation:
                            parts_list.append(scaled_variant * 2)
                            parts_list.append(scaled_variant * 2 + 1)
                        else:
                            parts_list.append(scaled_variant)
        return parts_list

    def _build_base_crop_specs(self) -> list[tuple[int, int, int, int]]:
        max_left = max(0, self._base_w - self._sample_x)
        max_top = max(0, self._base_h - self._sample_y)
        specs: list[tuple[int, int, int, int]] = []
        for location in range(self._base_locations):
            if self._random_crop:
                base_left = random.randint(0, max_left) if max_left > 0 else 0
                base_top = random.randint(0, max_top) if max_top > 0 else 0
            else:
                row, col = divmod(location, self._width_steps)
                base_left = col * self._step
                base_top = row * self._step

            specs.append((base_left, base_top, self._sample_x, self._sample_y))
        return specs

    def _build_scaled_crop_specs(self) -> list[tuple[int, int, int, int]]:
        if self._base_crop_specs is None:
            return []

        specs: list[tuple[int, int, int, int]] = []
        for base_left, base_top, _base_w, _base_h in self._base_crop_specs:
            crop_w = self._sample_x
            crop_h = self._sample_y
            if self._scale_augmentation_strength > 0.0:
                scale = random.uniform(
                    max(0.05, 1.0 - self._scale_augmentation_strength),
                    1.0 + self._scale_augmentation_strength,
                )
                crop_w = min(self._base_w, max(1, int(round(self._sample_x / scale))))
                crop_h = min(self._base_h, max(1, int(round(self._sample_y / scale))))

            center_x = base_left + (self._sample_x / 2.0)
            center_y = base_top + (self._sample_y / 2.0)
            left = int(round(center_x - (crop_w / 2.0)))
            top = int(round(center_y - (crop_h / 2.0)))
            left = min(max(0, left), max(0, self._base_w - crop_w))
            top = min(max(0, top), max(0, self._base_h - crop_h))
            specs.append((left, top, crop_w, crop_h))
        return specs

    def _resolve_crop_coordinates(self, location: int, scale_variant: int = 0) -> tuple[int, int, int, int]:
        crop_specs = self._scaled_crop_specs if scale_variant == 1 else self._base_crop_specs
        if crop_specs is not None and location < len(crop_specs):
            left, top, crop_w, crop_h = crop_specs[location]
            right = left + crop_w
            bottom = top + crop_h
            return left, top, right, bottom
        if self._random_crop:
            left = 0
            top = 0
        else:
            row, col = divmod(location, self._width_steps)
            left = col * self._step
            top = row * self._step

        right = left + self._sample_x
        bottom = top + self._sample_y
        return left, top, right, bottom

    def _extract_patch(
        self,
        matrix: np.ndarray,
        location: int,
        *,
        is_label: bool = False,
        scale_variant: int = 0,
    ) -> np.ndarray:
        left, top, right, bottom = self._resolve_crop_coordinates(location, scale_variant=scale_variant)

        if right <= self._base_w and bottom <= self._base_h:
            patch = matrix[:, top:bottom, left:right].copy()
        elif right > self._base_w and bottom > self._base_h:
            patch = matrix[:, -self._sample_y:, -self._sample_x:].copy()
        elif right > self._base_w:
            patch = matrix[:, top:bottom, -self._sample_x:].copy()
        else:
            patch = matrix[:, -self._sample_y:, left:right].copy()

        if patch.shape[1] != self._sample_y or patch.shape[2] != self._sample_x:
            resample = Image.Resampling.NEAREST if is_label else Image.Resampling.BILINEAR
            patch = self._resize_patch_tensor(patch, (self._sample_x, self._sample_y), resample=resample)
        return patch

    def _decode_part_index(self, item: int) -> tuple[int, str, int, int]:
        """Decode an item index into location and augmentation variants."""

        loc = self._parts_list[item]
        augmentation_variant = 0
        if self._additional_augmentation:
            loc, augmentation_variant = divmod(loc, 2)
        scale_variant = 0
        if self._scale_augmentation:
            loc, scale_variant = divmod(loc, 2)

        transform_count = max(1, len(self._transform_variants))
        location, transform_index = divmod(int(loc), transform_count)
        return int(location), str(self._transform_variants[int(transform_index)]), int(scale_variant), int(augmentation_variant)

    def resolve_part_coordinates(self, item: int) -> tuple[int, int, int, int]:
        """Resolve source-image crop coordinates for an item index."""

        location, _transform_variant, scale_variant, _augmentation_variant = self._decode_part_index(item)
        return self._resolve_crop_coordinates(location, scale_variant=scale_variant)

    def _is_uniform_label_location(self, location: int, *, scale_variant: int = 0) -> bool:
        label_patch = self._extract_patch(
            self.label_matrix,
            location,
            is_label=True,
            scale_variant=scale_variant,
        )
        binary_patch = label_patch > 0.5
        return bool(binary_patch.all() or (~binary_patch).all())

    def _is_rare_location(self, location: int, *, scale_variant: int = 0) -> bool:
        if self.rare_mask_matrix is None or self._rare_patch_oversampling_factor <= 1:
            return False
        rare_patch = self._extract_patch(
            self.rare_mask_matrix,
            location,
            is_label=True,
            scale_variant=scale_variant,
        )
        return bool(np.any(rare_patch > 0.5))

    def __getitem__(self, item):
        # if self._use_accelerator and _sample_fast_cutter_getitem is not None:
        #     try:
        #         image, label = _sample_fast_cutter_getitem(
        #             self._parts_list,
        #             item,
        #             self._vertical_rotation,
        #             self._horizontal_rotation,
        #             self._width_steps,
        #             self._step,
        #             self._sample_x,
        #             self._sample_y,
        #             self._base_w,
        #             self._base_h,
        #             self.image_matrix,
        #             self.label_matrix,
        #         )
        #         if self._additional_augmentation:
        #             image = self._apply_additional_augmentation(image)
        #         return image, label
        #     except Exception:
        #         # Disable accelerator for this instance after first failure.
        #         self._use_accelerator = False

        location, transform_variant, scale_variant, augmentation_variant = self._decode_part_index(item)

        image = self._extract_patch(self.image_matrix, location, scale_variant=scale_variant)
        label = self._extract_patch(self.label_matrix, location, is_label=True, scale_variant=scale_variant)

        if transform_variant != 'identity':
            image = _apply_transform_variant(image, transform_variant)
            label = _apply_transform_variant(label, transform_variant)

        if self._additional_augmentation and augmentation_variant == 1:
            image = self._apply_additional_augmentation(image)

        return image,label

    def _apply_additional_augmentation(self, image: np.ndarray) -> np.ndarray:
        # Lightweight photometric augmentation; labels remain unchanged.
        img = image.copy()
        if self._augmentation_blur_probability > 0.0 and self._augmentation_blur_radius > 0.0:
            if random.random() < self._augmentation_blur_probability:
                blur_radius = random.uniform(0.0, self._augmentation_blur_radius)
                if blur_radius > 0.0:
                    img = self._apply_gaussian_blur(img, blur_radius)
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
        if self._augmentation_gamma_strength > 0.0:
            gamma = random.uniform(
                max(0.1, 1.0 - self._augmentation_gamma_strength),
                1.0 + self._augmentation_gamma_strength,
            )
            img = np.power(np.clip(img, 0.0, 1.0), gamma).astype(np.float32, copy=False)
        if random.random() < self._augmentation_noise_probability and self._augmentation_noise_sigma > 0.0:
            img += np.random.normal(0.0, self._augmentation_noise_sigma, size=img.shape).astype(np.float32)
        np.clip(img, 0.0, 1.0, out=img)
        return img.astype(np.float32, copy=False)

    @staticmethod
    def _apply_gaussian_blur(image: np.ndarray, radius: float) -> np.ndarray:
        if radius <= 0.0:
            return image.astype(np.float32, copy=False)
        pil_image = SampleFastCutter._array_to_pil_image(image)
        blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=float(radius)))
        return SampleFastCutter._pil_to_float_array(blurred, channels=int(image.shape[0]))

    @staticmethod
    def _array_to_pil_image(image: np.ndarray) -> Image.Image:
        channels = int(image.shape[0])
        clipped = np.clip(image, 0.0, 1.0)
        if channels == 1:
            plane = np.round(clipped[0] * 255.0).astype(np.uint8)
            return Image.fromarray(plane, mode='L')
        if channels == 3:
            rgb = np.round(np.transpose(clipped, (1, 2, 0)) * 255.0).astype(np.uint8)
            return Image.fromarray(rgb, mode='RGB')
        plane = np.round(clipped[0] * 255.0).astype(np.uint8)
        return Image.fromarray(plane, mode='L')

    @staticmethod
    def _pil_to_float_array(image: Image.Image, *, channels: int) -> np.ndarray:
        if channels == 1:
            grayscale = np.asarray(image.convert('L'), dtype=np.float32)
            return grayscale[None, :, :] / 255.0

        if channels == 3:
            rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
            return np.transpose(rgb, (2, 0, 1)) / 255.0

        grayscale = np.asarray(image.convert('L'), dtype=np.float32)[None, :, :] / 255.0
        return np.repeat(grayscale, channels, axis=0)

    @staticmethod
    def _resize_patch_tensor(
        patch: np.ndarray,
        size: tuple[int, int],
        *,
        resample: int,
    ) -> np.ndarray:
        target_w, target_h = int(size[0]), int(size[1])
        channels = int(patch.shape[0])
        if channels == 1:
            plane = np.clip(patch[0] * 255.0, 0.0, 255.0).astype(np.uint8)
            resized = Image.fromarray(plane, mode='L').resize((target_w, target_h), resample=resample)
            return np.asarray(resized, dtype=np.float32)[None, :, :] / 255.0

        resized_channels: list[np.ndarray] = []
        for channel_index in range(channels):
            plane = np.clip(patch[channel_index] * 255.0, 0.0, 255.0).astype(np.uint8)
            resized = Image.fromarray(plane, mode='L').resize((target_w, target_h), resample=resample)
            resized_channels.append(np.asarray(resized, dtype=np.float32))
        return np.stack(resized_channels, axis=0) / 255.0

    @staticmethod
    def get_matrix_from_image(img, channels):
        matrix = np.asarray(img, dtype=np.float32)
        matrix /= 255.0
        if channels == 1:
            matrix = np.reshape(matrix, (channels, matrix.shape[0], matrix.shape[1]))
        else:
            matrix = matrix.transpose(2, 0, 1)

        return np.ascontiguousarray(matrix)

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
        transform_variants = _build_enabled_transform_variants(
            enable_rotate_180=bool(getattr(self._params, 'vertical_rotation', False)),
            enable_rotate_90=bool(getattr(self._params, 'horizontal_rotation', False)),
            enable_flip_x=bool(getattr(self._params, 'flip_x', False)),
            enable_flip_y=bool(getattr(self._params, 'flip_y', False)),
            square_patch=bool(self._params.x_size == self._params.y_size),
        )
        transform_count = max(1, len(transform_variants))
        location, transform_index = divmod(int(part_number), transform_count)
        return location, int(transform_index)


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
