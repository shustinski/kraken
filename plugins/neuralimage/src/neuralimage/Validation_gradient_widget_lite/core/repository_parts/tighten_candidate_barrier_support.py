"""repository: tighten candidate barrier support."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.bridge_cut_mask import (
    _boundary_cue_map,
    _connected_support_from_seed,
    _separation_core_mask,
    _thin_barrier_map_from_cues,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import _binary_dilate

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.make_polygon_candidate import _make_polygon_candidate

def _tighten_candidate_with_barrier_support(
    probability: np.ndarray,
    mask: np.ndarray,
    high_mask: np.ndarray,
    *,
    low_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> np.ndarray:
    """Tighten inflated small/medium candidate boundaries using thin barrier evidence."""

    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return mask_bool
    candidate = _make_polygon_candidate(probability, mask_bool, ('refine',))
    if candidate is None or candidate.area < 6:
        return mask_bool
    if float(max(candidate.aspect_ratio, candidate.elongation)) >= float(config.boundary_snap_min_aspect):
        return mask_bool
    pad = max(1, int(config.separation_roi_padding))
    x0, y0, width, height = candidate.bbox
    roi_x0 = max(0, x0 - pad)
    roi_y0 = max(0, y0 - pad)
    roi_x1 = min(mask_bool.shape[1], x0 + width + pad)
    roi_y1 = min(mask_bool.shape[0], y0 + height + pad)
    candidate_roi = np.asarray(mask_bool[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    high_roi = np.asarray(high_mask[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    core_roi = _separation_core_mask(
        candidate_roi,
        high_roi,
        min_area=max(1, int(config.separation_core_min_area)),
    )
    if not np.any(core_roi):
        return mask_bool
    boundary_cues_roi = _boundary_cue_map(
        np.asarray(probability[roi_y0:roi_y1, roi_x0:roi_x1], dtype=np.float32),
        candidate_roi,
        low_threshold=low_threshold,
        config=config,
    )
    thin_barrier_roi = _thin_barrier_map_from_cues(
        boundary_cues_roi,
        candidate_roi,
        threshold=max(float(config.separation_barrier_threshold), 0.56),
        dilate_radius=0,
    )
    if not np.any(thin_barrier_roi):
        return mask_bool
    blocked_roi = thin_barrier_roi & ~_binary_dilate(core_roi, 1)
    if not np.any(blocked_roi):
        return mask_bool
    tightened_roi = _connected_support_from_seed(candidate_roi & ~blocked_roi, core_roi)
    if not np.any(tightened_roi):
        return mask_bool
    retained_fraction = float(np.count_nonzero(tightened_roi) / max(1, np.count_nonzero(candidate_roi)))
    if retained_fraction < 0.68:
        return mask_bool
    tightened_mask = mask_bool.copy()
    tightened_mask[roi_y0:roi_y1, roi_x0:roi_x1] = tightened_roi
    return tightened_mask
