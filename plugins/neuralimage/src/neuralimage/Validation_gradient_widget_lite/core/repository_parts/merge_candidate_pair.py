"""repository: merge candidate pair."""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    POLYGON_CONFIDENCE_HYSTERESIS_FLOOR,
    POLYGON_CONFIDENCE_COMPLETION_MAJOR_SCALE,
    POLYGON_CONFIDENCE_SPILL_TRIM_DELTA,
)
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import _binary_dilate

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.make_polygon_candidate import _make_polygon_candidate

def _filter_bridge_region(
    probability: np.ndarray,
    bridge_mask: np.ndarray,
    first_support: np.ndarray,
    second_support: np.ndarray,
    *,
    floor_threshold: float,
    trim_delta: float,
) -> np.ndarray:
    """Keep only bridge pixels that have enough support to justify connecting two regions."""

    bridge_bool = np.asarray(bridge_mask, dtype=bool)
    if not np.any(bridge_bool):
        return bridge_bool
    first_bool = np.asarray(first_support, dtype=bool)
    second_bool = np.asarray(second_support, dtype=bool)
    prob = np.asarray(probability, dtype=np.float32)
    first_values = np.asarray(prob[first_bool], dtype=np.float32)
    second_values = np.asarray(prob[second_bool], dtype=np.float32)
    if first_values.size == 0 or second_values.size == 0:
        return np.zeros_like(bridge_bool, dtype=bool)
    reference_mean = min(
        float(np.mean(first_values, dtype=np.float64)),
        float(np.mean(second_values, dtype=np.float64)),
    )
    bridge_threshold = max(
        float(floor_threshold),
        min(reference_mean - float(trim_delta), reference_mean * 0.92),
    )
    keep = bridge_bool & (prob >= bridge_threshold)
    if not np.any(keep):
        return np.zeros_like(bridge_bool, dtype=bool)
    kept_fraction = float(np.count_nonzero(keep) / max(1, np.count_nonzero(bridge_bool)))
    if kept_fraction < 0.35:
        return np.zeros_like(bridge_bool, dtype=bool)
    keep_touch = np.asarray(_binary_dilate(keep, 1), dtype=bool)
    if not (np.any(keep_touch & first_bool) and np.any(keep_touch & second_bool)):
        return np.zeros_like(bridge_bool, dtype=bool)
    return np.asarray(keep, dtype=bool)

def _merge_candidate_pair(
    probability: np.ndarray,
    first: _PolygonConfidenceCandidate,
    second: _PolygonConfidenceCandidate,
    *,
    bridge_radius: int = 0,
    floor_threshold: float = POLYGON_CONFIDENCE_HYSTERESIS_FLOOR,
    trim_delta: float = POLYGON_CONFIDENCE_SPILL_TRIM_DELTA,
) -> _PolygonConfidenceCandidate | None:
    first_mask = np.asarray(first.mask, dtype=bool)
    second_mask = np.asarray(second.mask, dtype=bool)
    merged_mask = first_mask | second_mask
    gap_radius = max(0, int(bridge_radius))
    if gap_radius > 0 and not np.any(first_mask & second_mask):
        x0 = max(0, min(int(first.bbox[0]), int(second.bbox[0])) - gap_radius)
        y0 = max(0, min(int(first.bbox[1]), int(second.bbox[1])) - gap_radius)
        x1 = min(first_mask.shape[1], max(int(first.bbox[0] + first.bbox[2]), int(second.bbox[0] + second.bbox[2])) + gap_radius)
        y1 = min(first_mask.shape[0], max(int(first.bbox[1] + first.bbox[3]), int(second.bbox[1] + second.bbox[3])) + gap_radius)
        first_roi = first_mask[y0:y1, x0:x1]
        second_roi = second_mask[y0:y1, x0:x1]
        bridge_roi = _binary_dilate(first_roi, gap_radius) & _binary_dilate(second_roi, gap_radius)
        if np.any(bridge_roi):
            bridge_roi = _filter_bridge_region(
                np.asarray(probability[y0:y1, x0:x1], dtype=np.float32),
                bridge_roi,
                first_roi,
                second_roi,
                floor_threshold=float(floor_threshold),
                trim_delta=float(trim_delta),
            )
        if np.any(bridge_roi):
            merged_mask = merged_mask.copy()
            merged_mask[y0:y1, x0:x1] |= bridge_roi
    merged_bbox = (
        max(0, min(int(first.bbox[0]), int(second.bbox[0])) - gap_radius),
        max(0, min(int(first.bbox[1]), int(second.bbox[1])) - gap_radius),
        min(first_mask.shape[1], max(int(first.bbox[0] + first.bbox[2]), int(second.bbox[0] + second.bbox[2])) + gap_radius)
        - max(0, min(int(first.bbox[0]), int(second.bbox[0])) - gap_radius),
        min(first_mask.shape[0], max(int(first.bbox[1] + first.bbox[3]), int(second.bbox[1] + second.bbox[3])) + gap_radius)
        - max(0, min(int(first.bbox[1]), int(second.bbox[1])) - gap_radius),
    )
    return _make_polygon_candidate(
        probability,
        merged_mask,
        tuple(sorted(set(first.source_branches) | set(second.source_branches))),
        roi_bbox=merged_bbox,
    )

def _merge_search_margin(config: PolygonConfidencePipelineConfig) -> int:
    """Return the maximum bbox margin needed for merge neighborhood search."""

    base_gap = max(0, int(config.merge_distance))
    axis_gap = max(1, base_gap * max(1, int(POLYGON_CONFIDENCE_COMPLETION_MAJOR_SCALE)))
    return max(base_gap, axis_gap)

@lru_cache(maxsize=64)
def _merge_tile_size(search_margin: int) -> int:
    """Return a stable tile size for spatial candidate indexing."""

    return max(16, 2 * max(1, int(search_margin)) + 8)

def _candidate_neighbor_pairs(
    candidates: list[_PolygonConfidenceCandidate],
    config: PolygonConfidencePipelineConfig,
) -> list[tuple[int, int]]:
    """Return unique candidate pairs that are spatially close enough to merit exact merge checks."""

    if len(candidates) <= 1:
        return []
    search_margin = _merge_search_margin(config)
    tile_size = _merge_tile_size(search_margin)
    bins: dict[tuple[int, int], list[int]] = {}
    pairs: set[tuple[int, int]] = set()
    for index, candidate in enumerate(candidates):
        x0, y0, width, height = candidate.bbox
        x1 = x0 + width
        y1 = y0 + height
        gx0 = (max(0, int(x0) - search_margin)) // tile_size
        gy0 = (max(0, int(y0) - search_margin)) // tile_size
        gx1 = (max(0, int(x1) + search_margin - 1)) // tile_size
        gy1 = (max(0, int(y1) + search_margin - 1)) // tile_size
        for grid_y in range(gy0, gy1 + 1):
            for grid_x in range(gx0, gx1 + 1):
                cell = (grid_x, grid_y)
                for other_index in bins.get(cell, ()): 
                    if other_index == index:
                        continue
                    pair = (other_index, index) if other_index < index else (index, other_index)
                    pairs.add(pair)
                bins.setdefault(cell, []).append(index)
    return sorted(pairs)
