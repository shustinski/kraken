"""repository: extract large polygon candidates."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidenceDebugCandidate,
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.extract_branch_candidates import _append_debug_candidate

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import _label_components

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.make_polygon_candidate import _make_polygon_candidate

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.reconstruct_large_polygon_candidate import _reconstruct_large_polygon_candidate

def _extract_large_polygon_candidates(
    probability: np.ndarray,
    local_normalized_probability: np.ndarray,
    high_mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
    start_candidate_id: int = 1,
) -> tuple[list[_PolygonConfidenceCandidate], np.ndarray, tuple[PolygonConfidenceDebugCandidate, ...], int]:
    """Extract large polygon candidates from strong seeds before small-detail processing."""

    accepted: list[_PolygonConfidenceCandidate] = []
    debug_rows: list[PolygonConfidenceDebugCandidate] = []
    accepted_mask = np.zeros_like(np.asarray(high_mask, dtype=bool), dtype=bool)
    high_labels, high_count = _label_components(high_mask)
    candidate_id = int(start_candidate_id)
    for label_id in range(1, int(high_count) + 1):
        seed_component = high_labels == label_id
        seed_candidate = _make_polygon_candidate(probability, seed_component, ('large_polygon_seed',))
        if seed_candidate is None:
            continue
        reconstructed = _reconstruct_large_polygon_candidate(
            probability,
            local_normalized_probability,
            seed_component,
            low_threshold=low_threshold,
            strong_threshold=strong_threshold,
            config=config,
        )
        candidate = reconstructed or seed_candidate
        bbox_width = max(1, int(candidate.bbox[2]))
        bbox_height = max(1, int(candidate.bbox[3]))
        major_span = max(bbox_width, bbox_height)
        geometry_ok = (
            candidate.area >= max(1, int(config.large_polygon_min_area))
            and major_span >= max(1, int(config.large_polygon_min_major_span))
            and candidate.extent >= float(config.large_polygon_min_extent)
            and max(candidate.aspect_ratio, candidate.elongation) >= float(config.large_polygon_min_aspect_ratio)
        )
        notes: list[str] = []
        if candidate.area < max(1, int(config.large_polygon_min_area)):
            notes.append('large_area_below_min')
        if major_span < max(1, int(config.large_polygon_min_major_span)):
            notes.append('large_major_span_too_small')
        if candidate.extent < float(config.large_polygon_min_extent):
            notes.append('large_extent_too_small')
        if max(candidate.aspect_ratio, candidate.elongation) < float(config.large_polygon_min_aspect_ratio):
            notes.append('large_ratio_too_small')
        has_high_core = bool(np.any(np.asarray(high_mask, dtype=bool) & np.asarray(candidate.mask, dtype=bool)))
        accepted_flag = bool(geometry_ok and has_high_core)
        if not has_high_core:
            notes.append('large_missing_high_core')
        _append_debug_candidate(debug_rows, candidate_id, 'large_polygon', candidate, accepted=accepted_flag, notes=tuple(notes))
        if accepted_flag:
            accepted.append(candidate)
            accepted_mask |= np.asarray(candidate.mask, dtype=bool)
        candidate_id += 1
    return accepted, accepted_mask, tuple(debug_rows), candidate_id

if TYPE_CHECKING:
    from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_confidence_config import _PolygonConfidenceCandidate
