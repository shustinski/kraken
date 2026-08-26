from __future__ import annotations

from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry

from .geometry import compute_polygon_metrics
from .polygon import PolygonData, integer_points

_CONDUCTOR_CATEGORIES = {"conductor", "metal_border", "metal_wide_gradient"}


def offset_conductor_polygons(polygons: list[PolygonData], offset_px: float) -> list[PolygonData]:
    """Expand or shrink conductor polygons with shapely.buffer; other categories are cloned."""

    distance = float(offset_px)
    if abs(distance) < 1e-9:
        return [polygon.clone() for polygon in polygons]
    offset: list[PolygonData] = []
    for polygon in polygons:
        if polygon.is_hole or str(polygon.category) not in _CONDUCTOR_CATEGORIES or len(polygon.points) < 3:
            offset.append(polygon.clone())
            continue
        buffered = _buffer_ring(polygon.points, distance)
        if buffered is None:
            continue
        clone = polygon.clone()
        clone.points = integer_points(buffered)
        area, perimeter, bbox = compute_polygon_metrics(clone.points)
        clone.area = float(area)
        clone.perimeter = float(perimeter)
        clone.bbox = bbox
        offset.append(clone)
    return offset


def _buffer_ring(points: list[tuple[float, float]], distance: float) -> list[tuple[float, float]] | None:
    geometry = ShapelyPolygon(points)
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty:
        return None
    buffered: BaseGeometry = geometry.buffer(distance, join_style=2, mitre_limit=5.0)
    if buffered.is_empty:
        return None
    chosen = _largest_polygon(buffered)
    if chosen is None or chosen.is_empty or chosen.exterior is None:
        return None
    coords = list(chosen.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) < 3:
        return None
    return [(float(x_coord), float(y_coord)) for x_coord, y_coord in coords]


def _largest_polygon(geometry: BaseGeometry) -> ShapelyPolygon | None:
    if isinstance(geometry, ShapelyPolygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        parts = [part for part in geometry.geoms if isinstance(part, ShapelyPolygon) and not part.is_empty]
        if not parts:
            return None
        return max(parts, key=lambda part: float(part.area))
    return None
