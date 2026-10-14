"""repository: merge connected candidate groups."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.extract_branch_candidates import (
    _axis_gap_close,
    _candidate_iou,
    _touches_within_distance,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.merge_candidate_pair import _merge_candidate_pair

def _merge_connected_candidate_groups(
    probability: np.ndarray,
    candidates: list[_PolygonConfidenceCandidate],
    pair_indices: list[tuple[int, int]],
    config: PolygonConfidencePipelineConfig,
) -> list[_PolygonConfidenceCandidate]:
    """Merge all candidate groups connected by exact merge predicates for one round."""

    if not pair_indices:
        return list(candidates)

    parent = list(range(len(candidates)))

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _union(left: int, right: int) -> None:
        root_left = _find(left)
        root_right = _find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    merge_threshold = float(config.merge_iou_threshold)
    merge_distance = int(config.merge_distance)
    for index_a, index_b in pair_indices:
        first = candidates[index_a]
        second = candidates[index_b]
        overlap = _candidate_iou(first, second)
        if overlap >= merge_threshold:
            _union(index_a, index_b)
            continue
        if _touches_within_distance(first, second, merge_distance) or _axis_gap_close(first, second, config):
            _union(index_a, index_b)

    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(_find(index), []).append(index)

    if all(len(indices) == 1 for indices in groups.values()):
        return list(candidates)

    next_candidates: list[_PolygonConfidenceCandidate] = []
    for indices in groups.values():
        ordered = sorted(indices)
        if len(ordered) == 1:
            next_candidates.append(candidates[ordered[0]])
            continue
        merged_candidate = candidates[ordered[0]]
        for current_index in ordered[1:]:
            merged_candidate = _merge_candidate_pair(
                probability,
                merged_candidate,
                candidates[current_index],
                bridge_radius=max(1, merge_distance),
                floor_threshold=float(config.hysteresis_low_floor),
                trim_delta=float(config.spill_trim_delta),
            )
            if merged_candidate is None:
                merged_candidate = candidates[ordered[0]]
                break
        next_candidates.append(merged_candidate)
    return next_candidates
