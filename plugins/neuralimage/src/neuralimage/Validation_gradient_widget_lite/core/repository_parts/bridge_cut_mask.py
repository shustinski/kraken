"""repository: bridge cut mask."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import (
    _binary_dilate,
    _label_components,
    skeletonize,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.mask_structure import _thin_bridge_map

def _connected_support_from_seed(support_mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    """Keep only support components connected to the provided seed mask."""

    support_bool = np.asarray(support_mask, dtype=bool)
    seed_bool = np.asarray(seed_mask, dtype=bool) & support_bool
    if not np.any(support_bool) or not np.any(seed_bool):
        return np.zeros_like(support_bool, dtype=bool)
    labels, count = _label_components(support_bool)
    if count <= 0:
        return np.zeros_like(support_bool, dtype=bool)
    seed_labels = np.unique(labels[seed_bool])
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size <= 0:
        return np.zeros_like(support_bool, dtype=bool)
    return np.isin(labels, seed_labels, assume_unique=True)

def _separation_core_mask(
    candidate_mask: np.ndarray,
    high_mask: np.ndarray,
    *,
    min_area: int,
) -> np.ndarray:
    """Return stable seed cores for one candidate, removing tiny fragments."""

    candidate_bool = np.asarray(candidate_mask, dtype=bool)
    core_mask = candidate_bool & np.asarray(high_mask, dtype=bool)
    if not np.any(core_mask):
        return np.zeros_like(candidate_bool, dtype=bool)
    labels, count = _label_components(core_mask)
    if count <= 0:
        return np.zeros_like(candidate_bool, dtype=bool)
    area_counts = np.bincount(labels.ravel(), minlength=count + 1)
    valid_ids = np.flatnonzero(area_counts >= max(1, int(min_area)))
    valid_ids = valid_ids[valid_ids > 0]
    if valid_ids.size <= 0:
        return np.zeros_like(candidate_bool, dtype=bool)
    return np.isin(labels, valid_ids, assume_unique=True)

def _max_filter_map(array: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    local_radius = max(1, int(radius))
    if ndi is not None:
        size = 2 * local_radius + 1
        return np.asarray(ndi.maximum_filter(values, size=size, mode='nearest'), dtype=np.float32)
    padded = np.pad(values, local_radius, mode='edge')
    result = np.empty_like(values, dtype=np.float32)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            patch = padded[y:y + 2 * local_radius + 1, x:x + 2 * local_radius + 1]
            result[y, x] = float(np.max(patch))
    return result

def _boundary_cue_map(
    probability: np.ndarray,
    candidate_region: np.ndarray,
    *,
    low_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> np.ndarray:
    """Build boundary cues that combine valleys, local contrast, and transition uncertainty."""

    from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_transition_uncertainty_maps import _local_contrast_map, _polygon_transition_uncertainty_maps
    candidate_bool = np.asarray(candidate_region, dtype=bool)
    if not np.any(candidate_bool):
        return np.zeros_like(candidate_bool, dtype=np.float32)
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    local_contrast = _local_contrast_map(prob, radius=1)
    _width_map, _inverse_local_contrast, transition_uncertainty = _polygon_transition_uncertainty_maps(
        prob,
        candidate_bool,
        contrast_radius=1,
    )
    candidate_values = np.asarray(prob[candidate_bool], dtype=np.float32)
    candidate_low = float(np.quantile(candidate_values, 0.20)) if candidate_values.size else float(low_threshold)
    valley_reference = max(float(low_threshold), candidate_low)
    valley_scale = max(0.04, 1.0 - valley_reference)
    low_score = np.clip((valley_reference - prob) / valley_scale, 0.0, 1.0).astype(np.float32)
    cue = np.clip(
        float(config.separation_boundary_low_weight) * low_score
        + float(config.separation_boundary_contrast_weight) * local_contrast
        + float(config.separation_boundary_uncertainty_weight) * transition_uncertainty,
        0.0,
        1.0,
    ).astype(np.float32)
    cue[~candidate_bool] = 0.0
    return cue

def _thin_barrier_map_from_cues(
    boundary_cues: np.ndarray,
    candidate_region: np.ndarray,
    *,
    threshold: float,
    dilate_radius: int,
) -> np.ndarray:
    """Convert wide boundary cues into a thin separating barrier."""

    candidate_bool = np.asarray(candidate_region, dtype=bool)
    cue = np.asarray(boundary_cues, dtype=np.float32)
    if not np.any(candidate_bool):
        return np.zeros_like(candidate_bool, dtype=bool)
    ridge_zone = candidate_bool & (cue >= float(threshold))
    if not np.any(ridge_zone):
        return np.zeros_like(candidate_bool, dtype=bool)
    local_max = _max_filter_map(cue, radius=1)
    ridge_center = ridge_zone & (cue >= (local_max - 1e-6))
    skeleton = skeletonize(ridge_zone)
    thin = (ridge_center | skeleton) & candidate_bool
    radius = max(0, int(dilate_radius))
    if radius > 0:
        thin = _binary_dilate(thin, radius) & candidate_bool
    return np.asarray(thin, dtype=bool)

def _bridge_cut_mask(
    probability: np.ndarray,
    candidate_region: np.ndarray,
    boundary_cues: np.ndarray,
    seed_mask: np.ndarray,
    *,
    config: PolygonConfidencePipelineConfig,
) -> np.ndarray:
    """Break weak narrow bridges that connect multiple nearby seed regions."""

    candidate_bool = np.asarray(candidate_region, dtype=bool)
    seed_bool = np.asarray(seed_mask, dtype=bool) & candidate_bool
    if not np.any(candidate_bool) or not np.any(seed_bool):
        return np.zeros_like(candidate_bool, dtype=bool)
    seed_labels, seed_count = _label_components(seed_bool)
    if seed_count <= 1:
        return np.zeros_like(candidate_bool, dtype=bool)
    thin_bridge = np.asarray(_thin_bridge_map(candidate_bool) > 0.0, dtype=bool)
    if not np.any(thin_bridge):
        return np.zeros_like(candidate_bool, dtype=bool)
    bridge_mask = thin_bridge & (
        np.asarray(probability, dtype=np.float32) <= float(config.separation_bridge_probability_max)
    ) & (
        np.asarray(boundary_cues, dtype=np.float32) >= float(config.separation_bridge_barrier_threshold)
    )
    if not np.any(bridge_mask):
        return np.zeros_like(candidate_bool, dtype=bool)
    labels, count = _label_components(bridge_mask)
    if count <= 0:
        return np.zeros_like(candidate_bool, dtype=bool)
    result = np.zeros_like(candidate_bool, dtype=bool)
    for label_id in range(1, int(count) + 1):
        component = labels == label_id
        if not np.any(component):
            continue
        expanded = _binary_dilate(component, 1) & candidate_bool
        touched_seeds = np.unique(seed_labels[expanded])
        touched_seeds = touched_seeds[touched_seeds > 0]
        if touched_seeds.size >= 2:
            result |= component
    return result
