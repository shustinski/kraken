from __future__ import annotations

from ..domain import PolygonData, compute_polygon_metrics, integer_points

MIN_ANTIALIASING_GRADE = 1
MAX_ANTIALIASING_GRADE = 5


def normalize_antialiasing_grade(value: int | float | str | None) -> int:
    try:
        grade = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        grade = MIN_ANTIALIASING_GRADE
    return max(MIN_ANTIALIASING_GRADE, min(MAX_ANTIALIASING_GRADE, grade))


def _chaikin_closed_ring(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    smoothed: list[tuple[float, float]] = []
    for index, current in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        current_x, current_y = float(current[0]), float(current[1])
        next_x, next_y = float(next_point[0]), float(next_point[1])
        smoothed.append((0.75 * current_x + 0.25 * next_x, 0.75 * current_y + 0.25 * next_y))
        smoothed.append((0.25 * current_x + 0.75 * next_x, 0.25 * current_y + 0.75 * next_y))
    return smoothed


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
    """Return a corner-smoothed clone using repeated Chaikin corner cutting."""

    if polygon.shape_hint == "box" or polygon.category == "via" or len(polygon.points) < 3:
        return polygon.clone()
    points = [(float(x_coord), float(y_coord)) for x_coord, y_coord in polygon.points]
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    for _ in range(normalize_antialiasing_grade(grade)):
        points = _chaikin_closed_ring(points)
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
