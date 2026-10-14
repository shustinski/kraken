"""test_vector_geometry_postprocess: rect."""
from __future__ import annotations
from contour.domain import PolygonData, compute_polygon_metrics

def _rect(left: float, top: float, right: float, bottom: float, pid: int) -> PolygonData:
    pts = [
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
    ]
    a, per, bbox = compute_polygon_metrics(pts)
    return PolygonData(id=pid, points=pts, area=a, perimeter=per, bbox=bbox)
