"""repository: polygon transition uncertainty maps."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import _binary_dilate

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_confidence_pipeline import _polygon_confidence_pipeline

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.robust_normalize_display_map import _uncertainty_map_from_probability

def _polygon_confidence_support_mask(
    probability: np.ndarray,
    strong_mask: np.ndarray,
    *,
    strong_threshold: float = 0.5,
    config: PolygonConfidencePipelineConfig | None = None,
) -> np.ndarray:
    final_mask, _candidates, _debug = _polygon_confidence_pipeline(
        probability,
        strong_mask,
        strong_threshold=float(strong_threshold),
        config=config,
        include_debug=False,
    )
    return np.asarray(final_mask, dtype=bool)

def _local_contrast_map(probability: np.ndarray, radius: int = 1) -> np.ndarray:
    from neuralimage.Validation_gradient_widget_lite.core.repository import ndi

    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    local_radius = max(1, int(radius))
    if ndi is not None:
        size = 2 * local_radius + 1
        local_max = np.asarray(ndi.maximum_filter(prob, size=size, mode='nearest'), dtype=np.float32)
        local_min = np.asarray(ndi.minimum_filter(prob, size=size, mode='nearest'), dtype=np.float32)
        return np.clip(local_max - local_min, 0.0, 1.0).astype(np.float32)
    padded = np.pad(prob, local_radius, mode='edge')
    local_max = np.empty_like(prob, dtype=np.float32)
    local_min = np.empty_like(prob, dtype=np.float32)
    for y in range(prob.shape[0]):
        for x in range(prob.shape[1]):
            patch = padded[y:y + 2 * local_radius + 1, x:x + 2 * local_radius + 1]
            local_max[y, x] = float(np.max(patch))
            local_min[y, x] = float(np.min(patch))
    return np.clip(local_max - local_min, 0.0, 1.0).astype(np.float32)

def _local_mean_map(array: np.ndarray, radius: int = 1) -> np.ndarray:
    from neuralimage.Validation_gradient_widget_lite.core.repository import ndi

    values = np.asarray(array, dtype=np.float32)
    local_radius = max(1, int(radius))
    if ndi is not None:
        size = 2 * local_radius + 1
        return np.asarray(ndi.uniform_filter(values, size=size, mode='nearest'), dtype=np.float32)
    padded = np.pad(values, local_radius, mode='edge')
    result = np.empty_like(values, dtype=np.float32)
    patch_size = float((2 * local_radius + 1) ** 2)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            patch = padded[y:y + 2 * local_radius + 1, x:x + 2 * local_radius + 1]
            result[y, x] = float(np.sum(patch, dtype=np.float64) / patch_size)
    return result

def _polygon_transition_uncertainty_maps(probability: np.ndarray, support_mask: np.ndarray, *, contrast_radius: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    support = np.asarray(support_mask, dtype=bool)
    local_radius = max(1, int(contrast_radius))
    uncertainty_map = _uncertainty_map_from_probability(prob)
    local_contrast = _local_contrast_map(prob, radius=local_radius)
    width_raw = _local_mean_map(uncertainty_map, radius=local_radius)
    sample_mask = np.asarray(_binary_dilate(support, radius=local_radius), dtype=bool) if np.any(support) else np.ones_like(support, dtype=bool)

    def _scale(values: np.ndarray) -> float:
        valid = np.asarray(values, dtype=np.float32)
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            return 1.0
        scale = float(np.percentile(valid, 95.0))
        if scale <= EPS:
            scale = float(np.max(valid)) if valid.size else 1.0
        return max(scale, EPS)

    width_scale = _scale(width_raw[sample_mask])
    contrast_scale = _scale(local_contrast[sample_mask])
    transition_width_map = np.clip(width_raw / width_scale, 0.0, 1.0).astype(np.float32)
    inverse_local_contrast = (1.0 - np.clip(local_contrast / contrast_scale, 0.0, 1.0)).astype(np.float32)
    transition_uncertainty_map = np.clip(
        uncertainty_map * transition_width_map * (0.35 + 0.65 * inverse_local_contrast),
        0.0,
        1.0,
    ).astype(np.float32)
    return transition_width_map, inverse_local_contrast, transition_uncertainty_map
