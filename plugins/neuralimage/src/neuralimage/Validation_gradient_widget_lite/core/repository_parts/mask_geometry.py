"""repository: mask geometry."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import _binary_dilate_rect

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_transition_uncertainty_maps import _local_mean_map

def _normalize_probability_map(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0).astype(np.float32)

def _smooth_probability_map(probability: np.ndarray, config: PolygonConfidencePipelineConfig) -> np.ndarray:
    from neuralimage.Validation_gradient_widget_lite.core.repository import ndi

    prob = _normalize_probability_map(probability)
    if ndi is None:
        return prob
    result = prob
    median_radius = max(0, int(config.median_radius))
    if median_radius > 0:
        size = 2 * median_radius + 1
        result = np.asarray(ndi.median_filter(result, size=size, mode='nearest'), dtype=np.float32)
    sigma = max(0.0, float(config.gaussian_sigma))
    if sigma > EPS:
        result = np.asarray(ndi.gaussian_filter(result, sigma=sigma, mode='nearest'), dtype=np.float32)
    return _normalize_probability_map(result)

def _locally_normalized_probability_map(probability: np.ndarray, config: PolygonConfidencePipelineConfig) -> np.ndarray:
    prob = _normalize_probability_map(probability)
    radius = max(1, int(config.local_normalization_radius))
    mean_map = _local_mean_map(prob, radius=radius)
    sq_mean_map = _local_mean_map(np.square(prob, dtype=np.float32), radius=radius)
    variance_map = np.clip(sq_mean_map - np.square(mean_map, dtype=np.float32), 0.0, None)
    std_map = np.sqrt(variance_map + EPS, dtype=np.float32)
    normalized = np.clip(0.5 + 0.25 * ((prob - mean_map) / np.maximum(std_map, 0.05)), 0.0, 1.0)
    strength = float(max(0.0, min(1.0, config.local_normalization_strength)))
    enhanced = np.clip(prob + strength * (normalized - 0.5), 0.0, 1.0)
    return np.maximum(prob, np.asarray(enhanced, dtype=np.float32)).astype(np.float32)

def _preprocess_polygon_probability(probability: np.ndarray, config: PolygonConfidencePipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    prob = _normalize_probability_map(probability)
    smoothed = np.maximum(prob, _smooth_probability_map(prob, config))
    locally_normalized = _locally_normalized_probability_map(prob, config)
    preprocessed = np.maximum(smoothed, locally_normalized).astype(np.float32)
    return _normalize_probability_map(preprocessed), _normalize_probability_map(locally_normalized)

def _binary_erode_rect(mask: np.ndarray, radius_y: int = 1, radius_x: int = 1) -> np.ndarray:
    from neuralimage.Validation_gradient_widget_lite.core.repository import ndi

    mask_bool = np.asarray(mask, dtype=bool)
    ry = max(0, int(radius_y))
    rx = max(0, int(radius_x))
    if ry <= 0 and rx <= 0:
        return mask_bool.copy()
    if ndi is not None:
        structure = np.ones((2 * ry + 1, 2 * rx + 1), dtype=bool)
        return np.asarray(ndi.binary_erosion(mask_bool, structure=structure), dtype=bool)
    padded = np.pad(mask_bool, ((ry, ry), (rx, rx)), mode='constant', constant_values=False)
    result = np.ones_like(mask_bool, dtype=bool)
    for row_offset in range(2 * ry + 1):
        for column_offset in range(2 * rx + 1):
            result &= padded[row_offset:row_offset + mask_bool.shape[0], column_offset:column_offset + mask_bool.shape[1]]
    return result

def _binary_close_rect(mask: np.ndarray, radius_y: int = 1, radius_x: int = 1) -> np.ndarray:
    dilated = _binary_dilate_rect(mask, radius_y, radius_x)
    return _binary_erode_rect(dilated, radius_y, radius_x)

def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size == 0 or xs.size == 0:
        return 0, 0, 0, 0
    y0 = int(np.min(ys))
    y1 = int(np.max(ys)) + 1
    x0 = int(np.min(xs))
    x1 = int(np.max(xs)) + 1
    return x0, y0, x1 - x0, y1 - y0

def _mask_geometry(mask: np.ndarray) -> tuple[int, tuple[int, int, int, int], float, float, float]:
    mask_bool = np.asarray(mask, dtype=bool)
    area = int(np.count_nonzero(mask_bool))
    bbox = _mask_bbox(mask_bool)
    width = max(1, int(bbox[2]))
    height = max(1, int(bbox[3]))
    aspect_ratio = float(max(width, height) / max(1, min(width, height)))
    extent = float(area / max(1, width * height))
    elongation = aspect_ratio
    ys, xs = np.nonzero(mask_bool)
    if ys.size >= 3:
        coords = np.column_stack((ys.astype(np.float32), xs.astype(np.float32)))
        centered = coords - np.mean(coords, axis=0, keepdims=True)
        covariance = np.matmul(centered.T, centered) / max(1.0, float(coords.shape[0] - 1))
        eigvals = np.linalg.eigvalsh(covariance)
        if eigvals.size >= 2:
            major = float(max(eigvals[-1], EPS))
            minor = float(max(eigvals[0], EPS))
            elongation = float(np.sqrt(major / minor))
    return area, bbox, aspect_ratio, elongation, extent
