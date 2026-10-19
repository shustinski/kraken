"""repository: extract branch candidates."""
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Sequence
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    POLYGON_CONFIDENCE_COMPLETION_MAJOR_SCALE,
)
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidenceDebugCandidate,
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import (
    _binary_dilate,
    _label_components,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.make_polygon_candidate import _make_polygon_candidate

def _candidate_object_labels(
    candidates: Sequence[_PolygonConfidenceCandidate],
    shape: tuple[int, int],
) -> np.ndarray:
    """Build a label map that preserves final-candidate identity even when masks touch."""

    labels = np.zeros(shape, dtype=np.int32)
    for object_id, candidate in enumerate(candidates, start=1):
        candidate_mask = np.asarray(candidate.mask, dtype=bool)
        if candidate_mask.shape != shape or not np.any(candidate_mask):
            continue
        assignable = candidate_mask & (labels == 0)
        if np.any(assignable):
            labels[assignable] = int(object_id)
    return labels

def _append_debug_candidate(rows: list[PolygonConfidenceDebugCandidate], candidate_id: int, branch: str, candidate: _PolygonConfidenceCandidate, *, accepted: bool, notes: tuple[str, ...] = ()) -> None:
    rows.append(PolygonConfidenceDebugCandidate(
        object_id=int(candidate_id),
        branch=str(branch),
        source_branches=tuple(candidate.source_branches),
        accepted=bool(accepted),
        area=int(candidate.area),
        bbox_x=int(candidate.bbox[0]),
        bbox_y=int(candidate.bbox[1]),
        bbox_width=int(candidate.bbox[2]),
        bbox_height=int(candidate.bbox[3]),
        aspect_ratio=float(candidate.aspect_ratio),
        elongation=float(candidate.elongation),
        peak_probability=float(candidate.peak_probability),
        mean_probability=float(candidate.mean_probability),
        extent=float(candidate.extent),
        notes=tuple(str(note) for note in notes),
    ))

def _extract_branch_candidates(
    probability: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
    *,
    branch: str,
    min_area: int = 1,
    max_area: int | None = None,
    require_high_core: bool = True,
    acceptance_fn=None,
    start_candidate_id: int = 1,
) -> tuple[list[_PolygonConfidenceCandidate], np.ndarray, tuple[PolygonConfidenceDebugCandidate, ...], int]:
    accepted: list[_PolygonConfidenceCandidate] = []
    debug_rows: list[PolygonConfidenceDebugCandidate] = []
    accepted_mask = np.zeros_like(np.asarray(low_mask, dtype=bool), dtype=bool)
    labels, count = _label_components(low_mask)
    candidate_id = int(start_candidate_id)
    for label_id in range(1, int(count) + 1):
        component_mask = labels == label_id
        candidate = _make_polygon_candidate(probability, component_mask, (branch,))
        if candidate is None:
            continue
        has_high_core = bool(np.any(np.asarray(high_mask, dtype=bool) & component_mask))
        notes: list[str] = []
        accepted_flag = candidate.area >= max(1, int(min_area))
        if max_area is not None:
            accepted_flag = accepted_flag and candidate.area <= max(1, int(max_area))
            if candidate.area > max(1, int(max_area)):
                notes.append('area_above_max')
        if candidate.area < max(1, int(min_area)):
            notes.append('area_below_min')
        if require_high_core and not has_high_core:
            accepted_flag = False
            notes.append('missing_high_core')
        if acceptance_fn is not None:
            branch_ok, branch_notes = acceptance_fn(candidate, has_high_core)
            accepted_flag = accepted_flag and bool(branch_ok)
            notes.extend(tuple(branch_notes))
        _append_debug_candidate(debug_rows, candidate_id, branch, candidate, accepted=accepted_flag, notes=tuple(notes))
        if accepted_flag:
            accepted.append(candidate)
            accepted_mask |= candidate.mask
        candidate_id += 1
    return accepted, accepted_mask, tuple(debug_rows), candidate_id

def _candidate_iou(first: _PolygonConfidenceCandidate, second: _PolygonConfidenceCandidate) -> float:
    x0_a, y0_a, w_a, h_a = first.bbox
    x0_b, y0_b, w_b, h_b = second.bbox
    x1_a = x0_a + w_a
    y1_a = y0_a + h_a
    x1_b = x0_b + w_b
    y1_b = y0_b + h_b
    ix0 = max(x0_a, x0_b)
    iy0 = max(y0_a, y0_b)
    ix1 = min(x1_a, x1_b)
    iy1 = min(y1_a, y1_b)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = int(np.count_nonzero(first.mask[iy0:iy1, ix0:ix1] & second.mask[iy0:iy1, ix0:ix1]))
    if intersection <= 0:
        return 0.0
    union = int(first.area + second.area - intersection)
    return float(intersection / max(1, union))

def _bbox_gap(first_bbox: tuple[int, int, int, int], second_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x0_a, y0_a, w_a, h_a = first_bbox
    x0_b, y0_b, w_b, h_b = second_bbox
    x1_a = x0_a + w_a
    y1_a = y0_a + h_a
    x1_b = x0_b + w_b
    y1_b = y0_b + h_b
    gap_x = max(0, max(x0_b - x1_a, x0_a - x1_b))
    gap_y = max(0, max(y0_b - y1_a, y0_a - y1_b))
    return int(gap_y), int(gap_x)

def _touches_within_distance(first: _PolygonConfidenceCandidate, second: _PolygonConfidenceCandidate, distance: int) -> bool:
    gap = max(0, int(distance))
    if gap <= 0:
        iy, ix = _bbox_gap(first.bbox, second.bbox)
        if iy > 0 or ix > 0:
            return False
        x0 = max(int(first.bbox[0]), int(second.bbox[0]))
        y0 = max(int(first.bbox[1]), int(second.bbox[1]))
        x1 = min(int(first.bbox[0] + first.bbox[2]), int(second.bbox[0] + second.bbox[2]))
        y1 = min(int(first.bbox[1] + first.bbox[3]), int(second.bbox[1] + second.bbox[3]))
        if x1 <= x0 or y1 <= y0:
            return False
        return bool(np.any(first.mask[y0:y1, x0:x1] & second.mask[y0:y1, x0:x1]))
    gap_y, gap_x = _bbox_gap(first.bbox, second.bbox)
    if gap_y > gap or gap_x > gap:
        return False
    x0 = max(0, min(int(first.bbox[0]), int(second.bbox[0])) - gap)
    y0 = max(0, min(int(first.bbox[1]), int(second.bbox[1])) - gap)
    x1 = min(first.mask.shape[1], max(int(first.bbox[0] + first.bbox[2]), int(second.bbox[0] + second.bbox[2])) + gap)
    y1 = min(first.mask.shape[0], max(int(first.bbox[1] + first.bbox[3]), int(second.bbox[1] + second.bbox[3])) + gap)
    roi_first = np.asarray(first.mask[y0:y1, x0:x1], dtype=bool)
    roi_second = np.asarray(second.mask[y0:y1, x0:x1], dtype=bool)
    return bool(np.any(_binary_dilate(roi_first, gap) & roi_second))

def _axis_gap_close(first: _PolygonConfidenceCandidate, second: _PolygonConfidenceCandidate, config: PolygonConfidencePipelineConfig) -> bool:
    x0_a, y0_a, w_a, h_a = first.bbox
    x0_b, y0_b, w_b, h_b = second.bbox
    x1_a = x0_a + w_a
    x1_b = x0_b + w_b
    y1_a = y0_a + h_a
    y1_b = y0_b + h_b
    overlap_x = min(x1_a, x1_b) - max(x0_a, x0_b)
    overlap_y = min(y1_a, y1_b) - max(y0_a, y0_b)
    vertical_gap = max(0, max(y0_b - y1_a, y0_a - y1_b))
    horizontal_gap = max(0, max(x0_b - x1_a, x0_a - x1_b))
    major_gap = max(1, int(config.merge_distance) * max(1, int(POLYGON_CONFIDENCE_COMPLETION_MAJOR_SCALE)))

    def _dominant_axis(candidate: _PolygonConfidenceCandidate) -> str:
        bbox_width = max(1, int(candidate.bbox[2]))
        bbox_height = max(1, int(candidate.bbox[3]))
        if bbox_width == bbox_height:
            return "compact"
        return "horizontal" if bbox_width >= bbox_height else "vertical"

    first_axis = _dominant_axis(first)
    second_axis = _dominant_axis(second)
    if first_axis == "compact" and second_axis != "compact":
        first_axis = second_axis
    elif second_axis == "compact" and first_axis != "compact":
        second_axis = first_axis
    if first_axis != second_axis:
        return False
    if first_axis == "vertical":
        return overlap_x > 0 and vertical_gap <= major_gap
    if first_axis == "horizontal":
        return overlap_y > 0 and horizontal_gap <= major_gap
    return False

if TYPE_CHECKING:
    from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_confidence_config import _PolygonConfidenceCandidate
