from __future__ import annotations

import cv2
import numpy as np

from neuralimage.preprocessing.config import PreprocessingConfig


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    normalized = array.astype(np.float32)
    if normalized.max() <= 1.0:
        normalized = normalized * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _to_float01(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _normalize_odd_kernel(size: int) -> int:
    resolved = max(3, int(size))
    if resolved % 2 == 0:
        resolved += 1
    return resolved


class SemPreprocessingPipeline:
    """Deterministic preprocessing pipeline for SEM IC layout images."""

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.config.any_enabled():
            return _to_float01(image)

        working = _ensure_uint8(image)
        if self.config.background_subtraction:
            working = self._background_subtraction(working)
        if self.config.illumination_correction:
            working = self._illumination_correction(working)
        if self.config.scan_line_suppression:
            working = self._scan_line_suppression(working)
        if self.config.denoise:
            working = self._denoise(working)
        if self.config.percentile_normalization:
            working = self._percentile_normalization(working)
        if self.config.clahe:
            working = self._clahe(working)
        return _to_float01(working)

    def _percentile_normalization(self, image: np.ndarray) -> np.ndarray:
        low = float(np.percentile(image, self.config.percentile_low))
        high = float(np.percentile(image, self.config.percentile_high))
        if high <= low:
            return image
        normalized = (image.astype(np.float32) - low) / (high - low)
        return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)

    def _clahe(self, image: np.ndarray) -> np.ndarray:
        tile_x, tile_y = self.config.clahe_tile_grid_size
        clahe = cv2.createCLAHE(
            clipLimit=float(self.config.clahe_clip_limit),
            tileGridSize=(max(1, int(tile_x)), max(1, int(tile_y))),
        )
        return clahe.apply(image)

    def _illumination_correction(self, image: np.ndarray) -> np.ndarray:
        kernel = _normalize_odd_kernel(self.config.illumination_kernel_size)
        background = cv2.GaussianBlur(image, (kernel, kernel), 0)
        corrected = cv2.divide(image, background, scale=128)
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def _background_subtraction(self, image: np.ndarray) -> np.ndarray:
        kernel = _normalize_odd_kernel(self.config.background_blur_kernel)
        background = cv2.GaussianBlur(image, (kernel, kernel), 0)
        subtracted = cv2.subtract(image, background)
        return np.clip(subtracted, 0, 255).astype(np.uint8)

    def _scan_line_suppression(self, image: np.ndarray) -> np.ndarray:
        strength = float(min(max(self.config.scan_line_strength, 0.0), 1.0))
        if strength <= 0.0:
            return image
        float_image = image.astype(np.float32)
        row_means = float_image.mean(axis=1, keepdims=True)
        global_mean = float(float_image.mean())
        corrected = float_image - strength * (row_means - global_mean)
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        strength = float(max(1.0, self.config.denoise_strength))
        return cv2.fastNlMeansDenoising(image, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def apply_preprocessing(image: np.ndarray, config: PreprocessingConfig | None = None) -> np.ndarray:
    return SemPreprocessingPipeline(config).apply(image)
