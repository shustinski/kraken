"""serializers: polygons cv object."""
from __future__ import annotations
from contour.domain import PolygonData, compute_polygon_metrics

from contour.serializers_parts.cv_point_points import (
    _cv_object_id,
    _cv_point_points,
    _cv_points_from_ring,
    _cv_polygon_from_points,
)

def _polygons_from_cv_object(
    item: dict[str, object],
    fallback_id: int,
    generated_id_start: int,
) -> tuple[list[PolygonData], int]:
    geometry_type = str(item.get("type", "")).lower()
    if geometry_type == "point":
        points, default_category, default_shape = _cv_point_points(item)
        area, perimeter, bbox = compute_polygon_metrics(points)
        return [
            PolygonData(
                id=_cv_object_id(item, fallback_id),
                points=points,
                is_hole=False,
                parent_id=None,
                category=default_category,
                shape_hint=default_shape,
                area=area,
                perimeter=perimeter,
                bbox=bbox,
            )
        ], generated_id_start
    elif geometry_type == "polygon":
        raw_rings = item.get("coordinates", [])
        if not isinstance(raw_rings, (list, tuple)) or not raw_rings:
            return [], generated_id_start
    else:
        return [], generated_id_start

    parent_id = _cv_object_id(item, fallback_id)
    polygons: list[PolygonData] = []
    outer = _cv_polygon_from_points(
        polygon_id=parent_id,
        points=_cv_points_from_ring(raw_rings[0]),
        is_hole=False,
        parent_id=None,
    )
    if outer is None:
        return [], generated_id_start
    polygons.append(outer)
    next_id = generated_id_start
    for raw_hole in raw_rings[1:]:
        hole = _cv_polygon_from_points(
            polygon_id=next_id,
            points=_cv_points_from_ring(raw_hole),
            is_hole=True,
            parent_id=parent_id,
        )
        if hole is not None:
            polygons.append(hole)
            next_id += 1
    return polygons, next_id
