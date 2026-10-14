"""test_widget_interactions: oversampled rectangle polygon."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QApplication,
)
from contour.domain import PolygonData, compute_polygon_metrics

def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def _rectangle_polygon(left: int, top: int, right: int, bottom: int) -> PolygonData:
    points = [
        (float(left), float(top)),
        (float(right), float(top)),
        (float(right), float(bottom)),
        (float(left), float(bottom)),
    ]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(id=1, points=points, area=area, perimeter=perimeter, bbox=bbox)

def _oversampled_rectangle_polygon(left: int, top: int, right: int, bottom: int) -> PolygonData:
    mid_x = (left + right) / 2.0
    mid_y = (top + bottom) / 2.0
    points = [
        (float(left), float(top)),
        (mid_x, float(top)),
        (float(right), float(top)),
        (float(right), mid_y),
        (float(right), float(bottom)),
        (mid_x, float(bottom)),
        (float(left), float(bottom)),
        (float(left), mid_y),
    ]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(id=1, points=points, area=area, perimeter=perimeter, bbox=bbox)

def _net_outline_area(polygons: list[PolygonData]) -> float:
    """Subtract hole areas from roots (handles flat lists after CSG raster ops)."""

    outers = sum(p.area for p in polygons if not p.is_hole)
    holes = sum(p.area for p in polygons if p.is_hole)
    return outers - holes

def _all_points_within(polygons: list[PolygonData], left: float, top: float, right: float, bottom: float) -> bool:
    return all(left <= x <= right and top <= y <= bottom for polygon in polygons for x, y in polygon.points)
