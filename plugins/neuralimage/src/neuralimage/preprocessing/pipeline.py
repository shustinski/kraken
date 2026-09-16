from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from neuralimage.preprocessing.config import PreprocessingConfig


def to_float01(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.size == 0:
        return array.astype(np.float32)
    if np.issubdtype(array.dtype, np.integer):
        scale = float(np.iinfo(array.dtype).max)
        return np.clip(array.astype(np.float32) / scale, 0.0, 1.0)
    values = array.astype(np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError('SEM image contains no finite values.')
    if float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
        return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    # Float microscopy readers commonly expose the native integer range as
    # floats. Scale by that range without quantising to eight bits.
    maximum = float(finite.max())
    scale = 65535.0 if maximum > 255.0 else 255.0
    return np.clip(np.nan_to_num(values, nan=0.0) / scale, 0.0, 1.0)


def image_to_channel_first_float01(image: Image.Image | np.ndarray, channels: int) -> np.ndarray:
    """Convert PIL/NumPy input without quantising uint16 SEM grayscale data."""
    array = np.asarray(image)
    requested_channels = int(channels)
    if requested_channels not in {1, 3}:
        raise ValueError('SEM normalization supports one or three input channels.')
    if array.ndim == 2:
        plane = to_float01(array)
        if requested_channels == 1:
            return np.ascontiguousarray(plane[None, ...])
        return np.ascontiguousarray(np.repeat(plane[None, ...], 3, axis=0))
    if array.ndim != 3 or array.shape[2] < 1:
        raise ValueError(f'Unsupported SEM image shape: {array.shape!r}.')
    rgb = array[..., :3]
    if rgb.shape[2] == 1:
        plane = to_float01(rgb[..., 0])
        if requested_channels == 1:
            return np.ascontiguousarray(plane[None, ...])
        return np.ascontiguousarray(np.repeat(plane[None, ...], 3, axis=0))
    if requested_channels == 1:
        normalized_rgb = to_float01(rgb)
        plane = cv2.cvtColor(normalized_rgb.astype(np.float32), cv2.COLOR_RGB2GRAY)
        return np.ascontiguousarray(plane[None, ...])
    return np.ascontiguousarray(np.transpose(to_float01(rgb), (2, 0, 1)))


class SemPreprocessingPipeline:
    """Validated deterministic preprocessing in a float32 [0, 1] domain."""

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)

    def apply(self, image: np.ndarray) -> np.ndarray:
        working = to_float01(image)
        operations = {
            'background_subtraction': self._background_subtraction,
            'illumination_correction': self._illumination_correction,
            'scan_line_suppression': self._scan_line_suppression,
            'denoise': self._denoise,
            'percentile_normalization': self._percentile_normalization,
            'clahe': self._clahe,
        }
        for name in self.config.operation_order:
            if bool(getattr(self.config, name)):
                working = operations[name](working)
        normalizers = {
            'none': self._identity,
            'per_image_percentile': self._percentile_normalization,
            'dataset_zscore': self._dataset_zscore,
        }
        return normalizers[self.config.mode](working).astype(np.float32, copy=False)

    @staticmethod
    def _identity(image: np.ndarray) -> np.ndarray:
        return image

    def _percentile_normalization(self, image: np.ndarray) -> np.ndarray:
        low, high = np.percentile(image, (self.config.percentile_low, self.config.percentile_high))
        tolerance = max(1e-6, np.finfo(np.float32).eps * max(abs(float(low)), abs(float(high)), 1.0))
        if float(high - low) <= tolerance:
            return np.zeros_like(image, dtype=np.float32)
        return np.clip((image - float(low)) / float(high - low), 0.0, 1.0)

    def _dataset_zscore(self, image: np.ndarray) -> np.ndarray:
        if not self.config.has_dataset_statistics():
            raise ValueError('dataset_zscore requires mean/std calculated from the training dataset.')
        return (image - float(self.config.dataset_mean)) / float(self.config.dataset_std)

    def _clahe(self, image: np.ndarray) -> np.ndarray:
        # OpenCV CLAHE accepts uint16, preserving SEM detector precision.
        native = np.round(np.clip(image, 0.0, 1.0) * 65535.0).astype(np.uint16)
        clahe = cv2.createCLAHE(
            clipLimit=float(self.config.clahe_clip_limit),
            tileGridSize=tuple(int(value) for value in self.config.clahe_tile_grid_size),
        )
        return clahe.apply(native).astype(np.float32) / 65535.0

    def _illumination_correction(self, image: np.ndarray) -> np.ndarray:
        kernel = int(self.config.illumination_kernel_size)
        background = cv2.GaussianBlur(image, (kernel, kernel), 0)
        reference = max(float(np.median(background)), 1e-6)
        return np.clip(image * reference / np.maximum(background, 1e-6), 0.0, 1.0)

    def _background_subtraction(self, image: np.ndarray) -> np.ndarray:
        kernel = int(self.config.background_blur_kernel)
        background = cv2.GaussianBlur(image, (kernel, kernel), 0)
        return np.clip(image - background + float(np.median(background)), 0.0, 1.0)

    def _scan_line_suppression(self, image: np.ndarray) -> np.ndarray:
        strength = float(self.config.scan_line_strength)
        if strength <= 0.0:
            return image
        axis = 1 if self.config.scan_axis == 'rows' else 0
        profile = np.median(image, axis=axis).astype(np.float32)
        smooth = cv2.GaussianBlur(
            profile.reshape(-1, 1),
            (1, int(self.config.scan_profile_kernel)),
            0,
        ).reshape(-1)
        residual = profile - smooth
        correction = residual[:, None] if self.config.scan_axis == 'rows' else residual[None, :]
        return np.clip(image - strength * correction, 0.0, 1.0)

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        sigma_color = max(1e-4, float(self.config.denoise_strength) / 255.0)
        return cv2.bilateralFilter(image.astype(np.float32), d=5, sigmaColor=sigma_color, sigmaSpace=2.0)


def apply_preprocessing(image: np.ndarray, config: PreprocessingConfig | None = None) -> np.ndarray:
    return SemPreprocessingPipeline(config).apply(image)
