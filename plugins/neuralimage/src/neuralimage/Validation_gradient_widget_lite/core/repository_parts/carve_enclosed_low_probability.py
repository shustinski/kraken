"""repository: carve enclosed low probability."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import (
    _binary_dilate,
    _boundary_mask,
    _label_components,
)

def _carve_enclosed_low_probability_holes(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    low_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return mask_bool
    hole_threshold = min(float(config.hole_probability_max), max(float(config.hysteresis_low_floor), float(low_threshold) * float(config.hole_probability_scale)))
    low_inside = mask_bool & (np.asarray(probability, dtype=np.float32) <= hole_threshold)
    if not np.any(low_inside):
        return mask_bool
    hole_labels, hole_count = _label_components(low_inside)
    if hole_count <= 0:
        return mask_bool
    object_boundary = _boundary_mask(mask_bool)
    carved = mask_bool.copy()
    min_hole_area = max(1, int(config.hole_min_area))
    for label_id in range(1, int(hole_count) + 1):
        hole_mask = hole_labels == label_id
        hole_area = int(np.count_nonzero(hole_mask))
        if hole_area < min_hole_area:
            continue
        if np.any(hole_mask & object_boundary):
            continue
        surround = (_binary_dilate(hole_mask, 1) & mask_bool) & ~hole_mask
        if not np.any(surround):
            continue
        hole_mean = float(np.mean(np.asarray(probability[hole_mask], dtype=np.float32), dtype=np.float64))
        surround_mean = float(np.mean(np.asarray(probability[surround], dtype=np.float32), dtype=np.float64))
        if surround_mean <= hole_mean + 0.05:
            continue
        carved[hole_mask] = False
    return carved
