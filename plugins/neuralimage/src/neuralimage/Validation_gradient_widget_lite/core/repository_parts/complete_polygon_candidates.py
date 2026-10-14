"""repository: complete polygon candidates."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    POLYGON_CONFIDENCE_HYSTERESIS_FLOOR,
    POLYGON_CONFIDENCE_COMPLETION_BRIDGE_RADIUS,
    POLYGON_CONFIDENCE_SPILL_TRIM_DELTA,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import (
    _binary_dilate_rect,
    _completion_radii_for_mask,
    _label_components,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.make_polygon_candidate import _make_polygon_candidate

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.merge_candidate_pair import _filter_bridge_region

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_confidence_config import _polygon_confidence_completion_threshold

def _complete_polygon_candidates(
    probability: np.ndarray,
    candidates: list[_PolygonConfidenceCandidate],
    *,
    strong_threshold: float,
    weak_threshold: float,
    high_seed_mask: np.ndarray | None = None,
) -> list[_PolygonConfidenceCandidate]:
    if not candidates:
        return []
    completion_threshold = _polygon_confidence_completion_threshold(float(strong_threshold), float(weak_threshold))
    completion_mask = np.asarray(probability >= completion_threshold, dtype=bool)
    if not np.any(completion_mask):
        return list(candidates)
    completion_labels, completion_count = _label_components(completion_mask)
    if completion_count <= 0:
        return list(candidates)
    high_labels = None
    if high_seed_mask is not None:
        high_labels, _high_count = _label_components(np.asarray(high_seed_mask, dtype=bool) & completion_mask)
    bridge_radius = max(0, int(POLYGON_CONFIDENCE_COMPLETION_BRIDGE_RADIUS))
    completed: list[_PolygonConfidenceCandidate] = []
    for candidate in candidates:
        seed_mask = np.asarray(candidate.mask, dtype=bool)
        reach_radius_y, reach_radius_x = _completion_radii_for_mask(seed_mask, bridge_radius)
        reach_mask = _binary_dilate_rect(seed_mask, reach_radius_y, reach_radius_x) if (reach_radius_y > 0 or reach_radius_x > 0) else seed_mask
        grown_mask = np.asarray(seed_mask, dtype=bool)
        completion_ids = [int(candidate_id) for candidate_id in np.unique(completion_labels[reach_mask]) if int(candidate_id) > 0]
        for completion_id in completion_ids:
            component_mask = completion_labels == completion_id
            component_add = component_mask
            if high_labels is not None:
                component_high_ids = np.unique(high_labels[component_mask])
                component_high_ids = component_high_ids[component_high_ids > 0]
                if component_high_ids.size > 1:
                    current_high_ids = np.unique(high_labels[component_mask & seed_mask])
                    current_high_ids = current_high_ids[current_high_ids > 0]
                    if current_high_ids.size > 0:
                        valid_seed_mask = np.isin(high_labels, component_high_ids, assume_unique=True) & component_mask
                        if np.any(valid_seed_mask):
                            _distances, nearest_indices = ndi.distance_transform_edt(~valid_seed_mask, return_indices=True)
                            nearest_seed_labels = high_labels[tuple(nearest_indices)]
                            component_add = component_mask & np.isin(nearest_seed_labels, current_high_ids, assume_unique=True)
                            component_add &= (
                                np.asarray(probability, dtype=np.float32) >= max(
                                    float(POLYGON_CONFIDENCE_HYSTERESIS_FLOOR),
                                    float(weak_threshold) + 0.02,
                                )
                            ) | seed_mask
            grown_mask |= component_add
            if (reach_radius_y > 0 or reach_radius_x > 0) and not np.any(component_mask & seed_mask):
                bridge_path = _binary_dilate_rect(seed_mask, reach_radius_y, reach_radius_x) & _binary_dilate_rect(component_mask, reach_radius_y, reach_radius_x)
                bridge_path = _filter_bridge_region(
                    probability,
                    bridge_path,
                    seed_mask,
                    component_mask,
                    floor_threshold=max(float(weak_threshold), float(POLYGON_CONFIDENCE_HYSTERESIS_FLOOR)),
                    trim_delta=float(POLYGON_CONFIDENCE_SPILL_TRIM_DELTA),
                )
                if np.any(bridge_path):
                    grown_mask |= bridge_path
        completed_candidate = _make_polygon_candidate(probability, grown_mask, candidate.source_branches)
        completed.append(completed_candidate if completed_candidate is not None else candidate)
    return completed
