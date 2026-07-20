"""Repair self-intersecting polygon rings without approxPolyDP simplification."""

from __future__ import annotations

import cv2
import numpy as np

from .domain.polygon_ring import is_valid_closed_polygon_ring

_TOPOLOGY_REPAIR_MIN_FILL_IOU = 0.995
_COLLINEAR_ANGLE_EPS_DEG = 0.35


def _dense_contour_points(contour: np.ndarray) -> list[tuple[float, float]]:
    return [(float(p[0][0]), float(p[0][1])) for p in contour]


def _is_collinear_continuation(
    prev_point: tuple[float, float],
    current_point: tuple[float, float],
    next_point: tuple[float, float],
) -> bool:
    v1x = current_point[0] - prev_point[0]
    v1y = current_point[1] - prev_point[1]
    v2x = next_point[0] - current_point[0]
    v2y = next_point[1] - current_point[1]
    n1 = float(np.hypot(v1x, v1y))
    n2 = float(np.hypot(v2x, v2y))
    if n1 < 1e-6 or n2 < 1e-6:
        return True
    cross = abs(v1x * v2y - v1y * v2x)
    dot = v1x * v2x + v1y * v2y
    tolerance = float(np.sin(np.radians(_COLLINEAR_ANGLE_EPS_DEG))) * n1 * n2
    return dot >= 0.0 and cross <= tolerance


def _remove_collinear_polygon_vertices(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 4:
        return list(points)
    cleaned: list[tuple[float, float]] = []
    for point in points:
        while len(cleaned) >= 2 and _is_collinear_continuation(cleaned[-2], cleaned[-1], point):
            cleaned.pop()
        cleaned.append(point)

    # The stack handles every interior run. Only the cyclic seam can still
    # contain redundant vertices, so this loop is bounded by the seam run.
    while len(cleaned) >= 4:
        if _is_collinear_continuation(cleaned[-1], cleaned[0], cleaned[1]):
            del cleaned[0]
        elif _is_collinear_continuation(cleaned[-2], cleaned[-1], cleaned[0]):
            cleaned.pop()
        else:
            break
    return cleaned


def _localize_rings(
    rings: list[list[tuple[float, float]]],
    shape_hw: tuple[int, int],
    *,
    padding: int = 2,
) -> tuple[list[list[tuple[float, float]]], tuple[int, int], tuple[int, int]]:
    height, width = shape_hw
    coords = [point for ring in rings for point in ring]
    if not coords:
        return rings, shape_hw, (0, 0)
    min_x = max(0, int(np.floor(min(point[0] for point in coords))) - padding)
    min_y = max(0, int(np.floor(min(point[1] for point in coords))) - padding)
    max_x = min(width - 1, int(np.ceil(max(point[0] for point in coords))) + padding)
    max_y = min(height - 1, int(np.ceil(max(point[1] for point in coords))) + padding)
    local = [[(x - min_x, y - min_y) for x, y in ring] for ring in rings]
    return local, (max_y - min_y + 1, max_x - min_x + 1), (min_x, min_y)


def _filled_points_iou(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
    shape_hw: tuple[int, int],
) -> float:
    h, w = shape_hw
    if h <= 0 or w <= 0 or len(left) < 3 or len(right) < 3:
        return 0.0
    localized, (h, w), _offset = _localize_rings([left, right], shape_hw)
    left_mask = np.zeros((h, w), dtype=np.uint8)
    right_mask = np.zeros((h, w), dtype=np.uint8)
    left_arr = np.array(localized[0], dtype=np.int32).reshape(-1, 1, 2)
    right_arr = np.array(localized[1], dtype=np.int32).reshape(-1, 1, 2)
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

    if source_contour is not None and len(source_contour) >= 3:
        source_points = _dense_contour_points(np.asarray(source_contour).reshape(-1, 1, 2))
    elif points is not None and len(points) >= 3:
        source_points = list(points)
    else:
        return None

    localized, (local_h, local_w), (offset_x, offset_y) = _localize_rings([source_points], shape_hw)
    mask = np.zeros((local_h, local_w), dtype=np.uint8)
    if source_contour is not None and len(source_contour) >= 3:
        local_contour = np.asarray(localized[0], dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(mask, [local_contour], -1, 255, thickness=-1)
    elif points is not None and len(points) >= 3:
        arr = np.array(localized[0], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [arr], 255)

    candidate = _repair_ring_candidates_from_mask(mask)
    if candidate is None:
        return None
    candidate = [(x + offset_x, y + offset_y) for x, y in candidate]

    if require_fill_iou and source_contour is None:
        ref = reference_points if reference_points is not None else points
        if ref is not None and len(ref) >= 3:
            if _filled_points_iou(ref, candidate, shape_hw) < _TOPOLOGY_REPAIR_MIN_FILL_IOU:
                return None
    return candidate
