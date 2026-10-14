"""dataset: extract binary single channel."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PatchSampleRequest:
    index: int
    size_xy: tuple[int, int]

    def __post_init__(self) -> None:
        width, height = (int(value) for value in self.size_xy)
        if int(self.index) < 0:
            raise ValueError('patch request index must be non-negative')
        if width <= 0 or height <= 0:
            raise ValueError('patch request size must be positive')

def _unwrap_tech_augmented_mask(result: np.ndarray | tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    if isinstance(result, tuple):
        return np.asarray(result[1])
    return np.asarray(result)

def _extract_binary_single_channel(image: np.ndarray, *, binary_tolerance: float) -> np.ndarray | None:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        if not _is_binary_like_plane(array, tolerance=binary_tolerance):
            return None
        return array[None, :, :].astype(np.float32, copy=False)
    if array.ndim != 3 or array.shape[0] <= 0:
        return None
    single_channel = array[:1]
    if array.shape[0] > 1 and not np.allclose(
        array,
        np.repeat(single_channel, array.shape[0], axis=0),
        atol=1e-6,
    ):
        return None
    if not _is_binary_like_plane(single_channel[0], tolerance=binary_tolerance):
        return None
    return single_channel.astype(np.float32, copy=False)

def _broadcast_augmented_mask(template: np.ndarray, augmented_mask: np.ndarray) -> np.ndarray:
    template_array = np.asarray(template, dtype=np.float32)
    augmented = np.asarray(augmented_mask, dtype=np.float32)
    if augmented.ndim == 2:
        augmented = augmented[None, :, :]
    if template_array.ndim == 2:
        return augmented[0].astype(np.float32, copy=False)
    if template_array.ndim != 3:
        return augmented.astype(np.float32, copy=False)
    if template_array.shape[0] <= 1:
        return augmented[:1].astype(np.float32, copy=False)
    return np.repeat(augmented[:1], template_array.shape[0], axis=0).astype(np.float32, copy=False)

def _is_binary_like_plane(plane: np.ndarray, tolerance: float) -> bool:
    if plane.size == 0:
        return False
    distances = np.minimum(np.abs(plane), np.abs(plane - 1.0))
    return bool(np.mean(distances <= float(tolerance)) >= 0.98)
