from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageFilter

from lib.data_interfaces import TechAugmentationParameters, build_tech_augmentation_config


def _shift_array(array: np.ndarray, dy: int, dx: int, fill_value: int | bool = 0) -> np.ndarray:
    shifted = np.full(array.shape, fill_value, dtype=array.dtype)
    height, width = array.shape

    src_y0 = max(0, -int(dy))
    src_y1 = min(height, height - max(0, int(dy)))
    src_x0 = max(0, -int(dx))
    src_x1 = min(width, width - max(0, int(dx)))
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return shifted

    dst_y0 = max(0, int(dy))
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = max(0, int(dx))
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return shifted


def _binary_dilation(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = max(0, int(kernel_size))
    if radius == 0:
        return mask.copy()
    dilated = np.zeros(mask.shape, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dilated |= _shift_array(mask, dy, dx, fill_value=False)
    return dilated


def _binary_erosion(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = max(0, int(kernel_size))
    if radius == 0:
        return mask.copy()
    eroded = np.ones(mask.shape, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            eroded &= _shift_array(mask, dy, dx, fill_value=False)
    return eroded


def _binary_opening(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    return _binary_dilation(_binary_erosion(mask, kernel_size), kernel_size)


def _binary_closing(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    return _binary_erosion(_binary_dilation(mask, kernel_size), kernel_size)


def _neighbor_sum(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8, copy=False)
    total = np.zeros(mask.shape, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            total += _shift_array(mask_u8, dy, dx, fill_value=0)
    return total


def _mask_to_pil(mask: np.ndarray) -> Image.Image:
    payload = np.clip(mask.astype(np.float32, copy=False), 0.0, 1.0)
    return Image.fromarray(np.rint(payload * 255.0).astype(np.uint8), mode='L')


def _pil_to_float_mask(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert('L'), dtype=np.float32) / 255.0


def _resize_float_mask(mask: np.ndarray, size_xy: tuple[int, int]) -> np.ndarray:
    return _pil_to_float_mask(
        _mask_to_pil(mask).resize((int(size_xy[0]), int(size_xy[1])), resample=Image.Resampling.BILINEAR)
    )


def _generate_coarse_noise(shape: tuple[int, int], cell_size: int) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    resolved_cell = max(1, int(cell_size))
    grid_h = max(1, math.ceil(height / resolved_cell))
    grid_w = max(1, math.ceil(width / resolved_cell))
    coarse = np.random.random_sample((grid_h, grid_w)).astype(np.float32)
    expanded = np.repeat(np.repeat(coarse, resolved_cell, axis=0), resolved_cell, axis=1)
    return expanded[:height, :width]


def _extract_single_channel(mask: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(mask)
    if array.ndim == 2:
        plane = array
    elif array.ndim == 3 and array.shape[0] == 1:
        plane = array[0]
    elif array.ndim == 3 and array.shape[-1] == 1:
        plane = array[:, :, 0]
    else:
        raise ValueError('TechVariationAugmentor expects a 2D mask or a single-channel tensor.')

    plane = np.asarray(plane)
    max_value = float(np.max(plane)) if plane.size else 1.0
    threshold_value = float(threshold if max_value <= 1.0 else threshold * max(1.0, max_value))
    binary = plane >= threshold_value
    return array, binary.astype(bool, copy=False)


def _restore_mask_shape(template: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    array = np.asarray(template)
    max_value = float(np.max(array)) if array.size else 1.0
    scale_value = 1.0 if max_value <= 1.0 else max_value
    restored = binary_mask.astype(np.float32) * scale_value

    if array.ndim == 2:
        return restored.astype(array.dtype, copy=False)
    if array.ndim == 3 and array.shape[0] == 1:
        return restored.astype(array.dtype, copy=False)[None, :, :]
    if array.ndim == 3 and array.shape[-1] == 1:
        return restored.astype(array.dtype, copy=False)[:, :, None]
    raise ValueError('Unsupported mask shape for TechVariationAugmentor output restoration.')


class TechVariationAugmentor:
    """Generate technology-like geometric variations for binary metallization masks."""

    def __init__(self, config: TechAugmentationParameters | Mapping[str, Any] | None):
        self.config = build_tech_augmentation_config(config)
        self.available_ops: tuple[tuple[str, Callable[[np.ndarray], np.ndarray], float], ...] = (
            (
                'global_width',
                self.apply_global_width_variation,
                float(self.config.global_width.probability),
            ),
            (
                'scale_rethreshold',
                self.apply_scale_rethreshold,
                float(self.config.scale_rethreshold.probability),
            ),
            (
                'blur_threshold',
                self.apply_blur_threshold,
                float(self.config.blur_threshold.probability),
            ),
            (
                'boundary_aware',
                self.apply_boundary_aware_variation,
                float(self.config.boundary_aware.probability),
            ),
            (
                'local_morphology',
                self.apply_local_morphology,
                float(self.config.local_morphology.probability),
            ),
            (
                'gap_variation',
                self.apply_gap_open_close_variation,
                float(self.config.gap_variation.probability),
            ),
        )

    def __call__(self, mask: np.ndarray) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Apply 1-3 random technology-variation operators to a binary mask."""

        template, original_binary = _extract_single_channel(mask, self.config.binarization_threshold)
        original_binary = original_binary.astype(bool, copy=True)
        augmented = original_binary.copy()
        applied_any = False

        if self.config.enabled and original_binary.size > 0:
            min_ops = max(1, int(self.config.min_operations))
            max_ops = max(min_ops, int(self.config.max_operations))
            eligible_ops = [item for item in self.available_ops if float(item[2]) > 0.0]
            if eligible_ops:
                op_count = min(len(eligible_ops), random.randint(min_ops, max_ops))
            else:
                op_count = 0
            for _name, operation, probability in random.sample(eligible_ops, k=op_count):
                if random.random() > probability:
                    continue
                candidate = operation(augmented)
                if self._is_reasonable_variation(augmented, candidate):
                    augmented = candidate
                    applied_any = True

        if not applied_any:
            augmented = original_binary

        original_out = _restore_mask_shape(template, original_binary)
        augmented_out = _restore_mask_shape(template, augmented)
        if self.config.debug_return_pair:
            return original_out, augmented_out
        return augmented_out

    def apply_global_width_variation(self, mask: np.ndarray) -> np.ndarray:
        """Simulate global line-width drift with dilation or erosion."""

        cfg = self.config.global_width
        kernel_size = random.randint(int(cfg.kernel_size_range[0]), int(cfg.kernel_size_range[1]))
        if random.random() < float(cfg.erosion_probability):
            return _binary_erosion(mask, kernel_size)
        return _binary_dilation(mask, kernel_size)

    def apply_scale_rethreshold(self, mask: np.ndarray) -> np.ndarray:
        """Simulate process scaling by resize, back-resize and binarization."""

        cfg = self.config.scale_rethreshold
        height, width = mask.shape
        scale = random.uniform(float(cfg.scale_range[0]), float(cfg.scale_range[1]))
        scaled_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        resized = _resize_float_mask(mask.astype(np.float32), scaled_size)
        restored = _resize_float_mask(resized, (width, height))
        return restored >= float(cfg.threshold)

    def apply_blur_threshold(self, mask: np.ndarray) -> np.ndarray:
        """Simulate edge smoothing with Gaussian blur and re-thresholding."""

        cfg = self.config.blur_threshold
        blur_radius = random.uniform(float(cfg.blur_radius_range[0]), float(cfg.blur_radius_range[1]))
        if blur_radius <= 0.0:
            return mask.copy()
        blurred = _pil_to_float_mask(_mask_to_pil(mask.astype(np.float32)).filter(ImageFilter.GaussianBlur(blur_radius)))
        return blurred >= float(cfg.threshold)

    def apply_boundary_aware_variation(self, mask: np.ndarray) -> np.ndarray:
        """Perturb only a boundary band, leaving polygon interiors stable."""

        cfg = self.config.boundary_aware
        boundary = np.logical_xor(_binary_dilation(mask, 1), _binary_erosion(mask, 1))
        if not boundary.any():
            return mask.copy()

        band_width = random.randint(int(cfg.band_width_range[0]), int(cfg.band_width_range[1]))
        boundary_band = _binary_dilation(boundary, band_width)
        noise = _generate_coarse_noise(
            mask.shape,
            random.randint(int(cfg.noise_cell_size_range[0]), int(cfg.noise_cell_size_range[1])),
        )
        neighbors = _neighbor_sum(mask)

        add_mask = (
            boundary_band
            & ~mask
            & (noise >= (1.0 - float(cfg.add_probability)))
            & (neighbors >= int(cfg.min_addition_support))
        )
        remove_mask = (
            boundary_band
            & mask
            & (noise <= float(cfg.remove_probability))
            & (neighbors >= int(cfg.min_removal_support))
        )

        augmented = mask.copy()
        augmented[add_mask] = True
        augmented[remove_mask] = False

        smoothing_kernel = max(0, int(cfg.smoothing_kernel_size))
        if smoothing_kernel > 0:
            smoothed = _binary_closing(_binary_opening(augmented, smoothing_kernel), smoothing_kernel)
            augmented = np.where(boundary_band, smoothed, augmented)
        return augmented

    def apply_local_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Apply dilation or erosion only inside random local ROIs."""

        cfg = self.config.local_morphology
        augmented = mask.copy()
        height, width = mask.shape
        roi_count = random.randint(int(cfg.roi_count_range[0]), int(cfg.roi_count_range[1]))
        for _ in range(roi_count):
            roi_h = max(4, int(round(height * random.uniform(*cfg.roi_size_ratio_range))))
            roi_w = max(4, int(round(width * random.uniform(*cfg.roi_size_ratio_range))))
            roi_h = min(height, roi_h)
            roi_w = min(width, roi_w)
            top = random.randint(0, max(0, height - roi_h))
            left = random.randint(0, max(0, width - roi_w))
            patch = augmented[top:top + roi_h, left:left + roi_w]
            kernel_size = random.randint(int(cfg.kernel_size_range[0]), int(cfg.kernel_size_range[1]))
            if random.random() < float(cfg.erosion_probability):
                augmented[top:top + roi_h, left:left + roi_w] = _binary_erosion(patch, kernel_size)
            else:
                augmented[top:top + roi_h, left:left + roi_w] = _binary_dilation(patch, kernel_size)
        return augmented

    def apply_gap_open_close_variation(self, mask: np.ndarray) -> np.ndarray:
        """Close narrow gaps or remove thin bridges with morphology."""

        cfg = self.config.gap_variation
        kernel_size = random.randint(int(cfg.kernel_size_range[0]), int(cfg.kernel_size_range[1]))
        neighbors = _neighbor_sum(mask)

        if random.random() < float(cfg.opening_probability):
            opened = _binary_opening(mask, kernel_size)
            removable = mask & ~opened & (neighbors <= int(cfg.max_bridge_neighbor_count))
            augmented = mask.copy()
            augmented[removable] = False
            return augmented

        closed = _binary_closing(mask, kernel_size)
        addable = ~mask & closed & (neighbors >= int(cfg.min_gap_neighbor_count))
        augmented = mask.copy()
        augmented[addable] = True
        return augmented

    def _is_reasonable_variation(self, reference: np.ndarray, candidate: np.ndarray) -> bool:
        """Reject implausible topology changes before committing them."""

        if candidate.shape != reference.shape:
            return False
        if reference.any() and not candidate.any():
            return False
        if (~reference).any() and candidate.all():
            return False

        changed_ratio = float(np.mean(reference != candidate))
        if changed_ratio > float(self.config.max_changed_pixels_ratio):
            return False

        foreground_delta = abs(float(candidate.mean()) - float(reference.mean()))
        if foreground_delta > float(self.config.max_foreground_ratio_delta):
            return False

        return True
