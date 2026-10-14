"""vector_geometry_postprocess: drop triangle outer artifacts."""
from __future__ import annotations
from contour.domain import PolygonData, compute_polygon_metrics

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import drop_orphan_holes

def drop_triangle_outer_artifacts(
    polygons: list[PolygonData],
    enabled: bool,
    *,
    min_outer_area_px2: float = 0.0,
) -> list[PolygonData]:
    if not enabled:
        return polygons
    drop_ids: set[int] = set()
    threshold = float(min_outer_area_px2)
    for p in polygons:
        if p.is_hole or len(p.points) != 3 or p.category == "via" or p.shape_hint == "box":
            continue
        if p.shape_hint == "manual_outline":
            continue
        if threshold > 0.0:
            area_abs = abs(float(getattr(p, "area", 0.0) or 0.0))
            if area_abs <= 0.0:
                area_abs = abs(float(compute_polygon_metrics(p.points)[0]))
            if area_abs >= threshold:
                continue
        drop_ids.add(p.id)
    survivors = [p for p in polygons if p.id not in drop_ids and (p.parent_id is None or p.parent_id not in drop_ids)]
    return drop_orphan_holes(survivors)

def drop_triangle_outer_artifacts_scoped(
    polygons: list[PolygonData],
    affected_ids: set[int],
    enabled: bool,
    *,
    min_outer_area_px2: float = 0.0,
) -> list[PolygonData]:
    if not enabled or not affected_ids:
        return polygons
    drop_ids: set[int] = set()
    threshold = float(min_outer_area_px2)
    for polygon in polygons:
        if polygon.id not in affected_ids:
            continue
        if polygon.is_hole or len(polygon.points) != 3 or polygon.category == "via" or polygon.shape_hint == "box":
            continue
        if polygon.shape_hint == "manual_outline":
            continue
        if threshold > 0.0:
            area_abs = abs(float(getattr(polygon, "area", 0.0) or 0.0))
            if area_abs <= 0.0:
                area_abs = abs(float(compute_polygon_metrics(polygon.points)[0]))
            if area_abs >= threshold:
                continue
        drop_ids.add(polygon.id)
    if not drop_ids:
        return polygons
    survivors = [
        polygon
        for polygon in polygons
        if polygon.id not in drop_ids and (polygon.parent_id is None or polygon.parent_id not in drop_ids)
    ]
    return drop_orphan_holes(survivors)
