"""Repair self-intersecting polygon rings without approxPolyDP simplification."""

from __future__ import annotations

import cv2
import numpy as np

from .domain.polygon_ring import TOPOLOGY_CHECK_MAX_VERTICES, is_valid_closed_polygon_ring

_TOPOLOGY_REPAIR_MIN_FILL_IOU = 0.98
_COLLINEAR_ANGLE_EPS_DEG = 0.35


def _dense_contour_points(contour: np.ndarray) -> list[tuple[float, float]]:
    return [(float(p[0][0]), float(p[0][1])) for p in contour]


def _vertex_turn_angle_deg(
    prev_point: tuple[float, float],
    current_point: tuple[float, float],
    next_point: tuple[float, float],
) -> float:
    v1x = prev_point[0] - current_point[0]
    v1y = prev_point[1] - current_point[1]
    v2x = next_point[0] - current_point[0]
    v2y = next_point[1] - current_point[1]
    n1 = float(np.hypot(v1x, v1y))
    n2 = float(np.hypot(v2x, v2y))
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(np.arccos(dot)))


def _remove_collinear_polygon_vertices(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 4:
        return list(points)
    cleaned = list(points)
    changed = True
    while changed and len(cleaned) >= 4:
        changed = False
        for index in range(len(cleaned)):
            prev_point = cleaned[(index - 1) % len(cleaned)]
            current_point = cleaned[index]
            next_point = cleaned[(index + 1) % len(cleaned)]
            if _vertex_turn_angle_deg(prev_point, current_point, next_point) >= 180.0 - _COLLINEAR_ANGLE_EPS_DEG:
                del cleaned[index]
                changed = True
                break
    return cleaned


def _filled_points_iou(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
    shape_hw: tuple[int, int],
) -> float:
    h, w = shape_hw
    if h <= 0 or w <= 0 or len(left) < 3 or len(right) < 3:
        return 0.0
    left_mask = np.zeros((h, w), dtype=np.uint8)
    right_mask = np.zeros((h, w), dtype=np.uint8)
    left_arr = np.array(left, dtype=np.int32).reshape(-1, 1, 2)
    right_arr = np.array(right, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(left_mask, [left_arr], 255)
    cv2.fillPoly(right_mask, [right_arr], 255)
    inter = int(np.logical_and(left_mask > 0, right_mask > 0).sum())
    union = int(np.logical_or(left_mask > 0, right_mask > 0).sum())
    if union <= 0:
        return 0.0
    return float(inter / union)


def _candidate_from_contour(contour: np.ndarray) -> list[tuple[float, float]] | None:
    dense = _dense_contour_points(contour)
    if len(dense) < 3:
        return None
    compact = _remove_collinear_polygon_vertices(dense)
    if len(compact) >= 3 and is_valid_closed_polygon_ring(compact):
        return compact
    if len(dense) > TOPOLOGY_CHECK_MAX_VERTICES:
        return dense
    if len(compact) >= 3:
        return compact
    return dense


def _repair_ring_candidates_from_mask(mask: np.ndarray) -> list[tuple[float, float]] | None:
    for chain_mode in (cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE):
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, chain_mode)
        if not contours:
            continue
        contour = max(contours, key=lambda item: abs(float(cv2.contourArea(item))))
        candidate = _candidate_from_contour(contour)
        if candidate is not None:
            return candidate
    return None


def filled_polygon_iou(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
    shape_hw: tuple[int, int],
) -> float:
    return _filled_points_iou(left, right, shape_hw)


def repair_ring_from_filled_region(
    *,
    shape_hw: tuple[int, int],
    points: list[tuple[float, float]] | None = None,
    source_contour: np.ndarray | None = None,
    require_fill_iou: bool = True,
    reference_points: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]] | None:
    """Rebuild a simple closed ring as the external boundary of a filled region.

    Uses raster fill + contour tracing (no approxPolyDP).
    """
    h, w = shape_hw
    if h <= 0 or w <= 0:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    if source_contour is not None and len(source_contour) >= 3:
        cv2.drawContours(mask, [np.asarray(source_contour, dtype=np.int32)], -1, 255, thickness=-1)
    elif points is not None and len(points) >= 3:
        arr = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [arr], 255)
    else:
        return None

    candidate = _repair_ring_candidates_from_mask(mask)
    if candidate is None:
        return None

    if require_fill_iou and source_contour is None:
        ref = reference_points if reference_points is not None else points
        if ref is not None and len(ref) >= 3:
            if _filled_points_iou(ref, candidate, shape_hw) < _TOPOLOGY_REPAIR_MIN_FILL_IOU:
                return None
    return candidate
