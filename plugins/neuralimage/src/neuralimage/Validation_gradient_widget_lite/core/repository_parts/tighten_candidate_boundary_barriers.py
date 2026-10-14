"""repository: tighten candidate boundary barriers."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.bridge_cut_mask import (
    _boundary_cue_map,
    _bridge_cut_mask,
    _connected_support_from_seed,
    _separation_core_mask,
    _thin_barrier_map_from_cues,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import (
    _binary_dilate,
    _boundary_mask,
    _label_components,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.make_polygon_candidate import _make_polygon_candidate

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.mask_structure import _thin_bridge_map

def _tighten_candidate_with_boundary_barriers(
    probability: np.ndarray,
    mask: np.ndarray,
    high_mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
    source_branches: tuple[str, ...],
) -> np.ndarray:
    """Trim local boundary overgrowth while preserving core-supported geometry."""

    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return mask_bool
    candidate = _make_polygon_candidate(probability, mask_bool, source_branches)
    if candidate is None:
        return mask_bool
    if candidate.area < max(6, int(config.separation_core_min_area) * 3):
        return mask_bool

    pad = max(1, int(config.separation_roi_padding))
    x0, y0, width, height = candidate.bbox
    roi_x0 = max(0, x0 - pad)
    roi_y0 = max(0, y0 - pad)
    roi_x1 = min(mask_bool.shape[1], x0 + width + pad)
    roi_y1 = min(mask_bool.shape[0], y0 + height + pad)

    candidate_roi = np.asarray(mask_bool[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    if not np.any(candidate_roi):
        return mask_bool
    probability_roi = np.asarray(probability[roi_y0:roi_y1, roi_x0:roi_x1], dtype=np.float32)
    high_roi = np.asarray(high_mask[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    core_roi = _separation_core_mask(
        candidate_roi,
        high_roi,
        min_area=max(1, int(config.separation_core_min_area)),
    )
    if not np.any(core_roi):
        return mask_bool
    _seed_labels, seed_count = _label_components(core_roi)

    boundary_cues_roi = _boundary_cue_map(
        probability_roi,
        candidate_roi,
        low_threshold=low_threshold,
        config=config,
    )
    cue_values = np.asarray(boundary_cues_roi[candidate_roi], dtype=np.float32)
    if cue_values.size <= 0:
        return mask_bool
    tighten_threshold = min(
        0.95,
        max(
            float(config.separation_barrier_threshold) + 0.08,
            float(np.quantile(cue_values, 0.72)),
        ),
    )
    thin_barrier_roi = _thin_barrier_map_from_cues(
        boundary_cues_roi,
        candidate_roi,
        threshold=tighten_threshold,
        dilate_radius=0,
    )
    if not np.any(thin_barrier_roi):
        return mask_bool

    boundary_seed = thin_barrier_roi & _binary_dilate(_boundary_mask(candidate_roi), 1)
    edge_barrier_roi = (
        _connected_support_from_seed(thin_barrier_roi, boundary_seed)
        if np.any(boundary_seed)
        else np.zeros_like(candidate_roi, dtype=bool)
    )
    bridge_cut_roi = _bridge_cut_mask(
        probability_roi,
        candidate_roi,
        boundary_cues_roi,
        core_roi,
        config=config,
    )
    thin_bridge_roi = np.asarray(_thin_bridge_map(candidate_roi) > 0.0, dtype=bool)
    internal_conflict_roi = (
        _binary_dilate(thin_barrier_roi & _binary_dilate(thin_bridge_roi, 1), 1) & candidate_roi
        if seed_count > 1 and np.any(thin_bridge_roi)
        else np.zeros_like(candidate_roi, dtype=bool)
    )
    protected_roi = core_roi
    blocked_roi = (edge_barrier_roi | internal_conflict_roi | bridge_cut_roi) & ~protected_roi
    if not np.any(blocked_roi):
        return mask_bool

    tightened_roi = _connected_support_from_seed(candidate_roi & ~blocked_roi, core_roi)
    if not np.any(tightened_roi):
        return mask_bool
    original_area = int(np.count_nonzero(candidate_roi))
    tightened_area = int(np.count_nonzero(tightened_roi))
    retained_fraction = float(tightened_area / max(1, original_area))
    min_retained_fraction = max(0.72, float(config.boundary_snap_min_retained_fraction))
    if retained_fraction < min_retained_fraction:
        return mask_bool
    if tightened_area >= original_area:
        return mask_bool

    refined_mask = mask_bool.copy()
    refined_mask[roi_y0:roi_y1, roi_x0:roi_x1] = tightened_roi
    return refined_mask
