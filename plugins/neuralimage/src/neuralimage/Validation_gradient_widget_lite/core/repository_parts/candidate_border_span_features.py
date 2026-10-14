"""repository: candidate border span features."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import _clip01

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import (
    _binary_dilate,
    _boundary_mask,
    _label_components,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_confidence_config import _polygon_confidence_weak_threshold

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.split_polygon_candidate_by import _split_polygon_candidate_by_barriers

def _split_polygon_candidate_by_seeds(
    probability: np.ndarray,
    candidate: _PolygonConfidenceCandidate,
    high_mask: np.ndarray,
    config: PolygonConfidencePipelineConfig,
) -> tuple[_PolygonConfidenceCandidate, ...]:
    """Backward-compatible wrapper around the barrier-aware seed separation stage."""

    split_candidates, _debug_payload = _split_polygon_candidate_by_barriers(
        probability,
        candidate,
        high_mask,
        low_threshold=_polygon_confidence_weak_threshold(float(np.max(np.asarray(probability[candidate.mask], dtype=np.float32))) if np.any(candidate.mask) else 0.5, config),
        strong_threshold=float(np.max(np.asarray(probability[candidate.mask], dtype=np.float32))) if np.any(candidate.mask) else 0.5,
        config=config,
        include_debug=False,
    )
    return split_candidates

def _candidate_boundary_separation(probability: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return 0.0
    boundary = _boundary_mask(mask_bool)
    outer_ring = _binary_dilate(mask_bool, 1) & ~mask_bool
    if not np.any(boundary) or not np.any(outer_ring):
        return 0.0
    inner_mean = float(np.mean(np.asarray(probability[boundary], dtype=np.float32), dtype=np.float64))
    outer_mean = float(np.mean(np.asarray(probability[outer_ring], dtype=np.float32), dtype=np.float64))
    return float(_clip01(inner_mean - outer_mean))

def _candidate_axis_coverage(mask: np.ndarray) -> tuple[float, float]:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return 0.0, 0.0
    height, width = mask_bool.shape
    row_coverage = np.count_nonzero(mask_bool, axis=1).astype(np.float32) / max(1, width)
    col_coverage = np.count_nonzero(mask_bool, axis=0).astype(np.float32) / max(1, height)
    active_row = row_coverage[row_coverage > 0.0]
    active_col = col_coverage[col_coverage > 0.0]
    row_p90 = float(np.quantile(active_row, 0.9)) if active_row.size else 0.0
    col_p90 = float(np.quantile(active_col, 0.9)) if active_col.size else 0.0
    return row_p90, col_p90

def _candidate_border_span_features(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    *,
    cross_axis_max: float,
) -> dict[str, float | bool]:
    """Return border-touch and span metrics shared by spill rejection and final trimming."""

    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    x0, y0, width, height = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    x1 = x0 + width
    y1 = y0 + height
    touches_lr = x0 <= 0 and x1 >= image_width
    touches_tb = y0 <= 0 and y1 >= image_height
    span_x = float(width / max(1, image_width))
    span_y = float(height / max(1, image_height))
    thin_cross_axis = (
        (touches_lr and span_y <= float(cross_axis_max))
        or (touches_tb and span_x <= float(cross_axis_max))
    )
    row_coverage_p90, col_coverage_p90 = _candidate_axis_coverage(mask)
    axis_coverage = 0.0
    if touches_lr:
        axis_coverage = max(axis_coverage, row_coverage_p90)
    if touches_tb:
        axis_coverage = max(axis_coverage, col_coverage_p90)
    return {
        'touches_lr': bool(touches_lr),
        'touches_tb': bool(touches_tb),
        'span_x': float(span_x),
        'span_y': float(span_y),
        'thin_cross_axis': bool(thin_cross_axis),
        'row_coverage_p90': float(row_coverage_p90),
        'col_coverage_p90': float(col_coverage_p90),
        'axis_coverage': float(axis_coverage),
    }

def _candidate_axis_support_strength(
    candidate_mask: np.ndarray,
    support_mask: np.ndarray,
    *,
    touches_lr: bool,
    touches_tb: bool,
) -> tuple[float, float]:
    """Return major-axis support coverage and support area fraction for a candidate."""

    candidate_bool = np.asarray(candidate_mask, dtype=bool)
    support_bool = np.asarray(support_mask, dtype=bool) & candidate_bool
    candidate_area = int(np.count_nonzero(candidate_bool))
    if candidate_area <= 0 or not np.any(support_bool):
        return 0.0, 0.0
    row_coverage_p90, col_coverage_p90 = _candidate_axis_coverage(support_bool)
    axis_coverage = 0.0
    if touches_lr:
        axis_coverage = max(axis_coverage, row_coverage_p90)
    if touches_tb:
        axis_coverage = max(axis_coverage, col_coverage_p90)
    support_area_fraction = float(np.count_nonzero(support_bool) / max(1, candidate_area))
    return float(axis_coverage), float(support_area_fraction)

def _candidate_has_compact_core(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    core_threshold: float,
    max_aspect_ratio: float,
) -> bool:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return False
    core_mask = mask_bool & (np.asarray(probability, dtype=np.float32) >= float(core_threshold))
    if not np.any(core_mask):
        return False
    labels, count = _label_components(core_mask)
    for label_id in range(1, int(count) + 1):
        component_mask = labels == label_id
        area = int(np.count_nonzero(component_mask))
        if area < 4:
            continue
        ys, xs = np.nonzero(component_mask)
        if ys.size == 0 or xs.size == 0:
            continue
        width = int(np.max(xs) - np.min(xs) + 1)
        height = int(np.max(ys) - np.min(ys) + 1)
        aspect_ratio = float(max(width, height) / max(1, min(width, height)))
        if aspect_ratio <= float(max_aspect_ratio):
            return True
    return False

def _compact_core_mask(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    core_threshold: float,
    max_aspect_ratio: float,
    min_area: int = 4,
) -> np.ndarray:
    """Return compact high-probability cores inside a candidate."""

    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return np.zeros_like(mask_bool, dtype=bool)
    core_mask = mask_bool & (np.asarray(probability, dtype=np.float32) >= float(core_threshold))
    if not np.any(core_mask):
        return np.zeros_like(mask_bool, dtype=bool)
    labels, count = _label_components(core_mask)
    compact = np.zeros_like(mask_bool, dtype=bool)
    for label_id in range(1, int(count) + 1):
        component_mask = labels == label_id
        area = int(np.count_nonzero(component_mask))
        if area < max(1, int(min_area)):
            continue
        ys, xs = np.nonzero(component_mask)
        if ys.size == 0 or xs.size == 0:
            continue
        width = int(np.max(xs) - np.min(xs) + 1)
        height = int(np.max(ys) - np.min(ys) + 1)
        aspect_ratio = float(max(width, height) / max(1, min(width, height)))
        if aspect_ratio <= float(max_aspect_ratio):
            compact |= component_mask
    return compact

def _local_prominence_map(probability: np.ndarray, radius: int) -> np.ndarray:
    """Estimate local prominence over a broader neighborhood."""

    from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_transition_uncertainty_maps import _local_mean_map
    prob = np.asarray(probability, dtype=np.float32)
    local_mean = _local_mean_map(prob, radius=max(1, int(radius)))
    prominence = np.clip(prob - local_mean, 0.0, 1.0)
    return np.asarray(prominence, dtype=np.float32)
