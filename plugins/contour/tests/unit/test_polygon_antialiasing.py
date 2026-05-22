from __future__ import annotations

from contour.application.polygon_antialiasing import antialias_polygon, antialias_polygons, normalize_antialiasing_grade
from contour.domain import PolygonData, compute_polygon_metrics


def _square(polygon_id: int = 1) -> PolygonData:
    points = [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(id=polygon_id, points=points, area=area, perimeter=perimeter, bbox=bbox)


def _oversampled_rectangle(polygon_id: int = 1) -> PolygonData:
    points = [
        (0.0, 0.0),
        (2.0, 0.0),
        (4.0, 0.0),
        (8.0, 0.0),
        (8.0, 3.0),
        (8.0, 8.0),
        (4.0, 8.0),
        (0.0, 8.0),
        (0.0, 4.0),
    ]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(id=polygon_id, points=points, area=area, perimeter=perimeter, bbox=bbox)


def test_antialias_polygon_simplifies_vertices_like_epsilon() -> None:
    original = _oversampled_rectangle()

    smoothed = antialias_polygon(original, 2)

    assert len(smoothed.points) < len(original.points)
    assert smoothed.points[0] == (0.0, 0.0)
    assert smoothed.bbox == (0, 0, 9, 9)


def test_antialias_polygons_can_limit_to_selected_ids() -> None:
    first = _square(1)
    second = _oversampled_rectangle(2)
    result, changed = antialias_polygons([first, second], 1, only_ids={2})

    assert changed
    assert result[0].points == first.points
    assert len(result[1].points) < len(second.points)


def test_antialias_skips_via_boxes() -> None:
    via = _square()
    via.category = "via"
    via.shape_hint = "box"

    smoothed = antialias_polygon(via, 5)

    assert smoothed.points == via.points


def test_normalize_antialiasing_grade_bounds_values() -> None:
    assert normalize_antialiasing_grade(0) == 1
    assert normalize_antialiasing_grade(99) == 5
    assert normalize_antialiasing_grade("bad") == 1
