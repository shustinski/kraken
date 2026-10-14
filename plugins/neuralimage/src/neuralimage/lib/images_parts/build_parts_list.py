"""images: build parts list."""
from __future__ import annotations
import PIL.Image
import numpy as np
import random
from neuralimage.lib.data_interfaces import SampleGenerationSettings

from neuralimage.lib.images_parts.build_enabled_transform_variants import _apply_transform_variant


class BuildPartsListMixin:
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
                        if self._augmentation_variant_count > 0:
                            for aug_variant in range(self._augmentation_slots):
                                parts_list.append((scaled_variant * self._augmentation_slots) + aug_variant)
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
            if (
                self._scale_augmentation_strength > 0.0
                and random.random() < self._scale_augmentation_probability
            ):
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

    @staticmethod
    def _transformed_source_size(
        source_size_hw: tuple[int, int],
        transform_variant: str,
    ) -> tuple[int, int]:
        height, width = int(source_size_hw[0]), int(source_size_hw[1])
        if transform_variant in {'rotate_90', 'rotate_270'}:
            return width, height
        return height, width

    @staticmethod
    def _transform_bbox(
        coords_px: tuple[int, int, int, int],
        source_size_hw: tuple[int, int],
        transform_variant: str,
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = (int(value) for value in coords_px)
        height, width = int(source_size_hw[0]), int(source_size_hw[1])

        if transform_variant == 'rotate_180':
            return width - right, height - bottom, width - left, height - top
        if transform_variant == 'rotate_90':
            return top, width - right, bottom, width - left
        if transform_variant == 'rotate_270':
            return height - bottom, left, height - top, right
        if transform_variant == 'flip_x':
            return left, height - bottom, right, height - top
        if transform_variant == 'flip_y':
            return width - right, top, width - left, bottom
        return left, top, right, bottom

    def resolve_part_geometry(self, item: int) -> dict[str, object]:
        """Return source-frame geometry for a possibly transformed patch."""

        location, transform_variant, scale_variant, augmentation_variant = self._decode_part_index(item)
        raw_coords = self._resolve_crop_coordinates(location, scale_variant=scale_variant)
        raw_source_size_hw = (int(self._base_h), int(self._base_w))
        transformed_coords = self._transform_bbox(raw_coords, raw_source_size_hw, transform_variant)
        transformed_size_hw = self._transformed_source_size(raw_source_size_hw, transform_variant)
        return {
            'coords_px': tuple(int(value) for value in transformed_coords),
            'source_size_hw': tuple(int(value) for value in transformed_size_hw),
            'raw_coords_px': tuple(int(value) for value in raw_coords),
            'raw_source_size_hw': raw_source_size_hw,
            'transform_variant': str(transform_variant),
            'scale_variant': int(scale_variant),
            'augmentation_variant': int(augmentation_variant),
        }

    def resolve_transformed_image_matrix(self, item: int) -> np.ndarray:
        """Return the full image in the same geometric orientation as ``item``."""

        location, transform_variant, _scale_variant, _augmentation_variant = self._decode_part_index(item)
        _ = location
        if transform_variant == 'identity':
            return self.image_matrix
        return _apply_transform_variant(self.image_matrix, transform_variant)
