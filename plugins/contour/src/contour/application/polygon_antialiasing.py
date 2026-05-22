from __future__ import annotations

import cv2
import numpy as np

from ..domain import PolygonData, compute_polygon_metrics, integer_points

MIN_ANTIALIASING_GRADE = 1
MAX_ANTIALIASING_GRADE = 5


def normalize_antialiasing_grade(value: int | float | str | None) -> int:
    try:
        grade = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        grade = MIN_ANTIALIASING_GRADE
    return max(MIN_ANTIALIASING_GRADE, min(MAX_ANTIALIASING_GRADE, grade))


def _deduplicate_closed_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for point in points:
        normalized = (float(point[0]), float(point[1]))
        if not deduped or deduped[-1] != normalized:
            deduped.append(normalized)
    if len(deduped) >= 2 and deduped[0] == deduped[-1]:
        deduped.pop()
    return deduped


def _simplify_closed_ring(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    points = _deduplicate_closed_points(points)
    if len(points) < 3 or epsilon <= 0.0:
        return points
    contour = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
    simplified = cv2.approxPolyDP(contour, float(epsilon), True).reshape((-1, 2))
    simplified_points = [(float(x_coord), float(y_coord)) for x_coord, y_coord in simplified]
    simplified_points = _deduplicate_closed_points(simplified_points)
    return simplified_points if len(simplified_points) >= 3 else points


def _point_signature(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    return tuple((round(float(x_coord), 6), round(float(y_coord), 6)) for x_coord, y_coord in points)


def _polygon_signature(polygon: PolygonData) -> tuple[object, ...]:
    return (
        int(polygon.id),
        bool(polygon.is_hole),
        polygon.parent_id,
        str(polygon.category),
        str(polygon.shape_hint),
        _point_signature(polygon.points),
    )


def antialias_polygon(polygon: PolygonData, grade: int) -> PolygonData:
    """Return a clone simplified with approxPolyDP-style epsilon smoothing."""

    if polygon.shape_hint == "box" or polygon.category == "via" or len(polygon.points) < 3:
        return polygon.clone()
    points = [(float(x_coord), float(y_coord)) for x_coord, y_coord in polygon.points]
    points = _simplify_closed_ring(points, float(normalize_antialiasing_grade(grade)))
    clone = polygon.clone()
    clone.points = integer_points(points)
    clone.area, clone.perimeter, clone.bbox = compute_polygon_metrics(clone.points)
    return clone


def antialias_polygons(
    polygons: list[PolygonData],
    grade: int,
    *,
    only_ids: set[int] | None = None,
) -> tuple[list[PolygonData], bool]:
    target_ids = None if only_ids is None else {int(polygon_id) for polygon_id in only_ids}
    processed: list[PolygonData] = []
    for polygon in polygons:
        if target_ids is not None and int(polygon.id) not in target_ids:
            processed.append(polygon.clone())
        else:
            processed.append(antialias_polygon(polygon, grade))
    before = sorted(_polygon_signature(polygon) for polygon in polygons)
    after = sorted(_polygon_signature(polygon) for polygon in processed)
    return processed, before != after
