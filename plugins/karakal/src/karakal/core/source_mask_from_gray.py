"""Build a classical binary mask from a grayscale original frame.

Contour-inspired MVP: gradient barriers partition the frame; polarity selects
regions that look like polygons / metal pours. No Contour runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mask_primitives import _binary_dilate, _label_components
from .repository_shared import cv2, ndi

SOURCE_MASK_ALGO_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class SourceMaskConfig:
    """Tunable defaults for grayscale → source_mask reconstruction."""

    edge_percentile: float = 85.0
    min_region_area: int = 60
    polarity_mode: str = "brighter_inside"
    smooth_sigma: float = 1.0
    background_sigma: float = 8.0
    barrier_dilate_px: int = 1
    rim_band_px: int = 3


def build_source_mask(
    original: np.ndarray,
    *,
    config: SourceMaskConfig | None = None,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return a boolean mask reconstructed from a grayscale original."""

    settings = config or SourceMaskConfig()
    gray = _prepare_gray(original, target_shape=target_shape)
    if gray.size == 0 or gray.ndim != 2:
        return np.zeros_like(gray, dtype=bool) if gray.ndim == 2 else np.zeros((0, 0), dtype=bool)

    normalized = _robust_normalize(gray)
    if float(np.std(normalized)) < 1e-6:
        return np.zeros_like(normalized, dtype=bool)

    smoothed = _smooth(normalized, float(settings.smooth_sigma))
    barriers = _gradient_barriers(smoothed, edge_percentile=float(settings.edge_percentile))
    relief = _relief_field(smoothed, background_sigma=float(settings.background_sigma))
    ridge = relief > max(float(np.percentile(relief, 70.0)), 1e-4)
    trench = relief < min(float(np.percentile(relief, 30.0)), -1e-4)
    polarity = str(settings.polarity_mode or "brighter_inside").strip().lower()
    if polarity == "ridge_touch":
        # Closed-boundary style: walls are the edge dipole, not the fill itself.
        wall = barriers | ridge | trench
    else:
        # Brighter-inside: only gradient barriers, so bright fills stay as regions.
        wall = barriers
    if int(settings.barrier_dilate_px) > 0:
        wall = _binary_dilate(wall, radius=int(settings.barrier_dilate_px))

    interior = ~wall
    labels, count = _label_components(interior)
    if count <= 0:
        return np.zeros_like(gray, dtype=bool)

    areas = np.bincount(labels.ravel(), minlength=count + 1)
    min_area = max(1, int(settings.min_region_area))
    keep = np.zeros(count + 1, dtype=bool)

    if polarity == "ridge_touch":
        rim = max(1, int(settings.rim_band_px))
        ridge_band = _binary_dilate(ridge, radius=rim)
        trench_band = _binary_dilate(trench, radius=rim)
        inside = interior
        ridge_contact = np.bincount(labels[ridge_band & inside], minlength=count + 1)
        trench_contact = np.bincount(labels[trench_band & inside], minlength=count + 1)
        keep = (ridge_contact > trench_contact) & (areas >= min_area)
    else:
        # brighter_inside: keep regions brighter than the global interior mean.
        interior_mean = float(np.mean(smoothed[interior], dtype=np.float64)) if np.any(interior) else 0.0
        for label_id in range(1, count + 1):
            if int(areas[label_id]) < min_area:
                continue
            region = labels == label_id
            if float(np.mean(smoothed[region], dtype=np.float64)) >= interior_mean:
                keep[label_id] = True

    keep[0] = False
    if not np.any(keep):
        return np.zeros_like(gray, dtype=bool)
    return keep[labels]


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shapes must match: {a.shape} != {b.shape}")
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    if union <= 0:
        return 1.0
    return float(intersection / union)


def mask_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shapes must match: {a.shape} != {b.shape}")
    area_a = int(np.count_nonzero(a))
    area_b = int(np.count_nonzero(b))
    if area_a + area_b <= 0:
        return 1.0
    intersection = int(np.count_nonzero(a & b))
    return float((2.0 * intersection) / (area_a + area_b))


def _prepare_gray(original: np.ndarray, *, target_shape: tuple[int, int] | None) -> np.ndarray:
    gray = np.asarray(original)
    if gray.ndim == 3:
        gray = np.asarray(np.mean(gray, axis=2), dtype=np.float32)
    else:
        gray = np.asarray(gray, dtype=np.float32)
    if target_shape is not None and tuple(int(v) for v in gray.shape) != tuple(int(v) for v in target_shape):
        gray = _resize_gray(gray, (int(target_shape[0]), int(target_shape[1])))
    return gray


def _resize_gray(gray: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    height, width = int(target_shape[0]), int(target_shape[1])
    source = np.asarray(gray, dtype=np.float32)
    if source.shape == (height, width):
        return source
    if cv2 is not None:
        return np.asarray(cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA), dtype=np.float32)
    # Nearest-neighbor fallback without Qt.
    ys = (np.linspace(0, source.shape[0] - 1, height)).astype(np.int32)
    xs = (np.linspace(0, source.shape[1] - 1, width)).astype(np.int32)
    return source[ys][:, xs].astype(np.float32, copy=False)


def _robust_normalize(gray: np.ndarray) -> np.ndarray:
    values = np.asarray(gray, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = (float(item) for item in np.percentile(finite, (5.0, 95.0)))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)


def _smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.asarray(values, dtype=np.float32)
    if cv2 is not None:
        return np.asarray(cv2.GaussianBlur(np.asarray(values, dtype=np.float32), (0, 0), float(sigma)), dtype=np.float32)
    if ndi is not None and hasattr(ndi, "gaussian_filter"):
        return np.asarray(ndi.gaussian_filter(np.asarray(values, dtype=np.float32), sigma=float(sigma)), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def _gradient_barriers(values: np.ndarray, *, edge_percentile: float) -> np.ndarray:
    grad_x = np.zeros_like(values, dtype=np.float32)
    grad_y = np.zeros_like(values, dtype=np.float32)
    grad_x[:, :-1] = np.abs(values[:, 1:] - values[:, :-1])
    grad_y[:-1, :] = np.abs(values[1:, :] - values[:-1, :])
    gradient = np.hypot(grad_x, grad_y).astype(np.float32, copy=False)
    positive = gradient[gradient > 0.0]
    if positive.size == 0:
        return np.zeros_like(values, dtype=bool)
    threshold = float(np.percentile(positive, float(edge_percentile)))
    threshold = max(threshold, float(np.mean(positive) + 0.5 * np.std(positive)), 0.04)
    return np.asarray(gradient >= threshold, dtype=bool)


def _relief_field(values: np.ndarray, *, background_sigma: float) -> np.ndarray:
    background = _smooth(values, max(0.5, float(background_sigma)))
    return np.asarray(values, dtype=np.float32) - np.asarray(background, dtype=np.float32)


__all__ = [
    "SOURCE_MASK_ALGO_VERSION",
    "SourceMaskConfig",
    "build_source_mask",
    "mask_dice",
    "mask_iou",
]
