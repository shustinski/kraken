"""repository: split polygon candidate by."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.assign_candidate_region_seeds import _assign_candidate_region_to_seeds

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.bridge_cut_mask import (
    _boundary_cue_map,
    _bridge_cut_mask,
    _separation_core_mask,
    _thin_barrier_map_from_cues,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import _label_components

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.make_polygon_candidate import _make_polygon_candidate

def _split_polygon_candidate_by_barriers(
    probability: np.ndarray,
    candidate: _PolygonConfidenceCandidate,
    high_mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
    include_debug: bool = False,
) -> tuple[tuple[_PolygonConfidenceCandidate, ...], dict[str, np.ndarray] | None]:
    """Split one candidate via thin barriers and bridge cuts, then grow from stable seed cores."""

    if not bool(config.enable_watershed) or ndi is None:
        return (candidate,), None

    candidate_mask = np.asarray(candidate.mask, dtype=bool)
    if not np.any(candidate_mask):
        return (candidate,), None
    pad = max(1, int(config.separation_roi_padding))
    x0, y0, width, height = candidate.bbox
    roi_x0 = max(0, x0 - pad)
    roi_y0 = max(0, y0 - pad)
    roi_x1 = min(candidate_mask.shape[1], x0 + width + pad)
    roi_y1 = min(candidate_mask.shape[0], y0 + height + pad)
    candidate_roi = np.asarray(candidate_mask[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    if not np.any(candidate_roi):
        return (candidate,), None
    probability_roi = np.asarray(probability[roi_y0:roi_y1, roi_x0:roi_x1], dtype=np.float32)
    high_roi = np.asarray(high_mask[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    core_roi = _separation_core_mask(
        candidate_roi,
        high_roi,
        min_area=max(1, int(config.separation_core_min_area)),
    )
    seed_labels, seed_count = _label_components(core_roi)
    debug_payload = None
    if include_debug:
        debug_payload = {
            "candidate_region": np.zeros_like(candidate_mask, dtype=bool),
            "core_seeds": np.zeros_like(candidate_mask, dtype=bool),
            "thin_barrier": np.zeros_like(candidate_mask, dtype=bool),
            "bridge_cuts": np.zeros_like(candidate_mask, dtype=bool),
            "barrier_blocked": np.zeros_like(candidate_mask, dtype=bool),
            "boundary_cues": np.zeros_like(probability, dtype=np.float32),
        }
        debug_payload["candidate_region"][roi_y0:roi_y1, roi_x0:roi_x1] = candidate_roi
        debug_payload["core_seeds"][roi_y0:roi_y1, roi_x0:roi_x1] = core_roi
    if seed_count <= 1:
        return (candidate,), debug_payload

    boundary_cues_roi = _boundary_cue_map(
        probability_roi,
        candidate_roi,
        low_threshold=low_threshold,
        config=config,
    )
    thin_barrier_roi = _thin_barrier_map_from_cues(
        boundary_cues_roi,
        candidate_roi,
        threshold=float(config.separation_barrier_threshold),
        dilate_radius=int(config.separation_barrier_dilate_radius),
    )
    bridge_cut_roi = _bridge_cut_mask(
        probability_roi,
        candidate_roi,
        boundary_cues_roi,
        core_roi,
        config=config,
    )
    if not np.any(thin_barrier_roi) and not np.any(bridge_cut_roi):
        return (candidate,), debug_payload
    blocked_roi = (thin_barrier_roi | bridge_cut_roi) & ~core_roi
    assigned_labels_roi = _assign_candidate_region_to_seeds(candidate_roi, seed_labels, blocked_roi)
    split_candidates: list[_PolygonConfidenceCandidate] = []
    for seed_id in range(1, int(seed_count) + 1):
        component_roi = assigned_labels_roi == seed_id
        if not np.any(component_roi):
            continue
        component_mask = np.zeros_like(candidate_mask, dtype=bool)
        component_mask[roi_y0:roi_y1, roi_x0:roi_x1] = component_roi
        component_candidate = _make_polygon_candidate(
            probability,
            component_mask,
            candidate.source_branches,
            roi_bbox=(roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0),
        )
        if component_candidate is not None:
            split_candidates.append(component_candidate)
    if include_debug and debug_payload is not None:
        debug_payload["thin_barrier"][roi_y0:roi_y1, roi_x0:roi_x1] = thin_barrier_roi
        debug_payload["bridge_cuts"][roi_y0:roi_y1, roi_x0:roi_x1] = bridge_cut_roi
        debug_payload["barrier_blocked"][roi_y0:roi_y1, roi_x0:roi_x1] = blocked_roi
        debug_payload["boundary_cues"][roi_y0:roi_y1, roi_x0:roi_x1] = boundary_cues_roi
    return tuple(split_candidates) if split_candidates else (candidate,), debug_payload
