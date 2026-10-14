"""images: sized item."""
from __future__ import annotations
from PIL import Image
import numpy as np

from neuralimage.lib.images_parts.build_enabled_transform_variants import _apply_transform_variant


class SizedItemMixin:
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
        if self._augmentation_variant_count > 0:
            loc, augmentation_variant = divmod(loc, self._augmentation_slots)
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

        if self._augmentation_variant_count > 0 and augmentation_variant > 0:
            image = self._apply_additional_augmentation(image)

        return image,label

    @staticmethod
    def _centered_bounds(
        *,
        center_x: float,
        center_y: float,
        width: int,
        height: int,
        source_width: int,
        source_height: int,
    ) -> tuple[int, int, int, int]:
        left = int(round(float(center_x) - (int(width) / 2.0)))
        top = int(round(float(center_y) - (int(height) / 2.0)))
        if source_width >= width:
            left = min(max(0, left), source_width - width)
        else:
            left = 0
        if source_height >= height:
            top = min(max(0, top), source_height - height)
        else:
            top = 0
        return left, top, left + int(width), top + int(height)

    def get_sized_item(
        self,
        item: int,
        *,
        size_xy: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        """Extract a physical source crop with a batch-selected output size."""

        width, height = (int(value) for value in size_xy)
        if width <= 0 or height <= 0:
            raise ValueError('requested patch size must be positive')
        _location, transform_variant, _scale_variant, augmentation_variant = self._decode_part_index(item)
        geometry = self.resolve_part_geometry(item)
        base_left, base_top, base_right, base_bottom = (
            int(value) for value in geometry['coords_px']
        )
        image_source = self.resolve_transformed_image_matrix(item)
        label_source = (
            self.label_matrix
            if transform_variant == 'identity'
            else _apply_transform_variant(self.label_matrix, transform_variant)
        )
        source_height, source_width = int(image_source.shape[-2]), int(image_source.shape[-1])
        bounds = self._centered_bounds(
            center_x=(base_left + base_right) / 2.0,
            center_y=(base_top + base_bottom) / 2.0,
            width=width,
            height=height,
            source_width=source_width,
            source_height=source_height,
        )
        left, top, right, bottom = bounds
        clipped_right = min(source_width, right)
        clipped_bottom = min(source_height, bottom)
        image = image_source[:, top:clipped_bottom, left:clipped_right].copy()
        label = label_source[:, top:clipped_bottom, left:clipped_right].copy()
        if image.shape[-2:] != (height, width):
            image = self._resize_patch_tensor(
                image,
                (width, height),
                resample=Image.Resampling.BILINEAR,
            )
            label = self._resize_patch_tensor(
                label,
                (width, height),
                resample=Image.Resampling.NEAREST,
            )
        if self._augmentation_variant_count > 0 and augmentation_variant > 0:
            image = self._apply_additional_augmentation(image)
        return image, label, bounds
