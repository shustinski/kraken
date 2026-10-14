"""repository: point local contrast."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    POINT_CONFIDENCE_NEIGHBOR_RADIUS,
)

def _point_confidence_patch(array: np.ndarray, x: float, y: float, radius: int) -> np.ndarray:
    patch_radius = max(0, int(radius))
    px = int(round(float(x)))
    py = int(round(float(y)))
    if array.ndim != 2 or array.size == 0 or py < 0 or py >= array.shape[0] or px < 0 or px >= array.shape[1]:
        return np.zeros((0, 0), dtype=np.float32)
    y0 = max(0, py - patch_radius)
    y1 = min(array.shape[0], py + patch_radius + 1)
    x0 = max(0, px - patch_radius)
    x1 = min(array.shape[1], px + patch_radius + 1)
    return np.asarray(array[y0:y1, x0:x1], dtype=np.float32)

def _point_local_contrast(probability: np.ndarray, x: float, y: float, radius: int = POINT_CONFIDENCE_NEIGHBOR_RADIUS) -> float:
    prob = np.asarray(probability, dtype=np.float32)
    px = int(round(float(x)))
    py = int(round(float(y)))
    if prob.ndim != 2 or prob.size == 0 or py < 0 or py >= prob.shape[0] or px < 0 or px >= prob.shape[1]:
        return 0.0
    patch = _point_confidence_patch(prob, x, y, radius)
    if patch.size <= 1:
        return 0.0
    center_value = float(prob[py, px])
    patch_sum = float(np.sum(patch, dtype=np.float64)) - center_value
    neighbor_count = max(1, int(patch.size - 1))
    return float(center_value - patch_sum / float(neighbor_count))
