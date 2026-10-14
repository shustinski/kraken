"""images: construction."""
from __future__ import annotations
import numpy as np
import random
from neuralimage.lib.data_interfaces import SampleGenerationSettings

from neuralimage.lib.images_parts.build_enabled_transform_variants import (
    _build_enabled_transform_variants,
    is_sample_fast_cutter_accelerated,
    resolved_augmentation_variant_count,
)

from neuralimage.lib.images_parts.sample_calculator import SampleCalculator


class ConstructionMixin:
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
        self._transform_probabilities = {
            'rotate_90': self._probability(parameters, 'horizontal_rotation_probability', 1.0),
            'rotate_270': self._probability(parameters, 'horizontal_rotation_probability', 1.0),
            'rotate_180': self._probability(parameters, 'vertical_rotation_probability', 1.0),
            'flip_x': self._probability(parameters, 'flip_x_probability', 1.0),
            'flip_y': self._probability(parameters, 'flip_y_probability', 1.0),
        }
        self._skip_uniform_labels = bool(skip_uniform_labels)
        self._rare_patch_oversampling_factor = max(1, int(rare_patch_oversampling_factor))
        self._random_crop = bool(getattr(parameters, 'random_crop', False))
        self._scale_augmentation = bool(getattr(parameters, 'scale_augmentation', False))
        self._scale_augmentation_probability = self._probability(
            parameters, 'scale_augmentation_probability', 1.0
        )
        self._scale_augmentation_strength = max(
            0.0, float(getattr(parameters, 'scale_augmentation_strength', 0.2))
        )
        self._additional_augmentation = bool(getattr(parameters, 'additional_augmentation', False))
        self._augmentation_variant_count = resolved_augmentation_variant_count(parameters)
        self._augmentation_slots = self._augmentation_variant_count + 1
        self._augmentation_brightness_strength = max(
            0.0, float(getattr(parameters, 'augmentation_brightness_strength', 0.1))
        )
        self._augmentation_brightness_enabled = bool(
            getattr(parameters, 'augmentation_brightness_enabled', True)
        )
        self._augmentation_brightness_probability = self._probability(
            parameters, 'augmentation_brightness_probability', 1.0
        )
        self._augmentation_contrast_strength = max(
            0.0, float(getattr(parameters, 'augmentation_contrast_strength', 0.1))
        )
        self._augmentation_contrast_enabled = bool(
            getattr(parameters, 'augmentation_contrast_enabled', True)
        )
        self._augmentation_contrast_probability = self._probability(
            parameters, 'augmentation_contrast_probability', 1.0
        )
        self._augmentation_gamma_strength = max(
            0.0, float(getattr(parameters, 'augmentation_gamma_strength', 0.15))
        )
        self._augmentation_gamma_enabled = bool(
            getattr(parameters, 'augmentation_gamma_enabled', True)
        )
        self._augmentation_gamma_probability = self._probability(
            parameters, 'augmentation_gamma_probability', 1.0
        )
        self._augmentation_noise_enabled = bool(getattr(parameters, 'augmentation_noise_enabled', True))
        self._augmentation_noise_probability = float(getattr(parameters, 'augmentation_noise_probability', 0.5))
        self._augmentation_noise_probability = min(1.0, max(0.0, self._augmentation_noise_probability))
        self._augmentation_noise_sigma = max(0.0, float(getattr(parameters, 'augmentation_noise_sigma', 0.01)))
        self._augmentation_blur_probability = min(
            1.0,
            max(0.0, float(getattr(parameters, 'augmentation_blur_probability', 0.25))),
        )
        self._augmentation_blur_enabled = bool(getattr(parameters, 'augmentation_blur_enabled', True))
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
        self._transform_variants = [
            variant
            if variant == 'identity' or random.random() < self._transform_probabilities.get(variant, 1.0)
            else 'identity'
            for variant in self._transform_variants
        ]
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
            and self._augmentation_variant_count <= 0
            and self.image_matrix.shape[0] == 1
            and (self._square_patch or not self._horizontal_rotation)
        )
