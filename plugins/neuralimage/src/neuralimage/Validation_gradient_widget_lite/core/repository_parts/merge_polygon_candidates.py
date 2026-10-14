"""repository: merge polygon candidates."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.merge_candidate_pair import _candidate_neighbor_pairs

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.merge_connected_candidate_groups import _merge_connected_candidate_groups

def _merge_polygon_candidates(
    probability: np.ndarray,
    candidates: list[_PolygonConfidenceCandidate],
    config: PolygonConfidencePipelineConfig,
) -> list[_PolygonConfidenceCandidate]:
    """Merge polygon candidates using spatial shortlists instead of repeated all-pairs scans."""

    merged = list(candidates)
    while True:
        pair_indices = _candidate_neighbor_pairs(merged, config)
        if not pair_indices:
            return merged
        next_candidates = _merge_connected_candidate_groups(probability, merged, pair_indices, config)
        if len(next_candidates) == len(merged):
            return merged
        merged = next_candidates
