"""images: apply additional augmentation."""
from __future__ import annotations
from PIL import Image, ImageFilter
import numpy as np
import random

from neuralimage.lib.images_parts.build_enabled_transform_variants import _build_enabled_transform_variants


class ApplyAdditionalAugmentationMixin:
    def _apply_additional_augmentation(self, image: np.ndarray) -> np.ndarray:
        # Lightweight photometric augmentation; labels remain unchanged.
        img = image.copy()
        if self._augmentation_blur_enabled and self._augmentation_blur_probability > 0.0 and self._augmentation_blur_radius > 0.0:
            if random.random() < self._augmentation_blur_probability:
                blur_radius = random.uniform(0.0, self._augmentation_blur_radius)
                if blur_radius > 0.0:
                    img = self._apply_gaussian_blur(img, blur_radius)
        if self._augmentation_brightness_enabled and random.random() < self._augmentation_brightness_probability:
            brightness = random.uniform(
                1.0 - self._augmentation_brightness_strength,
                1.0 + self._augmentation_brightness_strength,
            )
            img *= brightness
        if self._augmentation_contrast_enabled and random.random() < self._augmentation_contrast_probability:
            contrast = random.uniform(
                1.0 - self._augmentation_contrast_strength,
                1.0 + self._augmentation_contrast_strength,
            )
            mean = img.mean(axis=(1, 2), keepdims=True)
            img = (img - mean) * contrast + mean
        if (
            self._augmentation_gamma_enabled
            and self._augmentation_gamma_strength > 0.0
            and random.random() < self._augmentation_gamma_probability
        ):
            gamma = random.uniform(
                max(0.1, 1.0 - self._augmentation_gamma_strength),
                1.0 + self._augmentation_gamma_strength,
            )
            img = np.power(np.clip(img, 0.0, 1.0), gamma).astype(np.float32, copy=False)
        if self._augmentation_noise_enabled and random.random() < self._augmentation_noise_probability and self._augmentation_noise_sigma > 0.0:
            img += np.random.normal(0.0, self._augmentation_noise_sigma, size=img.shape).astype(np.float32)
        np.clip(img, 0.0, 1.0, out=img)
        return img.astype(np.float32, copy=False)

    @staticmethod
    def _probability(parameters: object, name: str, default: float) -> float:
        return min(1.0, max(0.0, float(getattr(parameters, name, default))))

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
