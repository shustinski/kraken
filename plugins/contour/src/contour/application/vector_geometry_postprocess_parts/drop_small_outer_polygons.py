"""vector_geometry_postprocess: drop small outer polygons."""
from __future__ import annotations
from typing import TYPE_CHECKING
from contour.domain import PolygonData

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import drop_orphan_holes

def should_defer_overlap_merge_after_edit(
    settings: VectorGeometrySettings,
    *,
    changed_polygon_ids: set[int],
    removed_ids: set[int],
) -> bool:
    """Whether overlap merge may run after commit instead of blocking the edit."""

    if not removed_ids:
        return False
    affected = len(changed_polygon_ids | removed_ids)
    threshold = max(1, int(settings.deferred_geometry_repair_affected_threshold))
    return affected >= threshold

def should_defer_geometry_repair(
    settings: VectorGeometrySettings,
    *,
    changed_polygon_ids: set[int],
    removed_ids: set[int],
) -> bool:
    """Backward-compatible alias for overlap-merge deferral."""

    return should_defer_overlap_merge_after_edit(
        settings,
        changed_polygon_ids=changed_polygon_ids,
        removed_ids=removed_ids,
    )

def drop_small_outer_polygons(polygons: list[PolygonData], min_area_px2: float) -> list[PolygonData]:
    if min_area_px2 <= 0.0:
        return polygons
    def _keep_outer(poly: PolygonData) -> bool:
        if poly.is_hole:
            return True
        if poly.category == "via" or poly.shape_hint == "box":
            return True
        return abs(float(poly.area)) >= float(min_area_px2)

    drop_ids = {p.id for p in polygons if not _keep_outer(p)}
    survivors: list[PolygonData] = []
    for p in polygons:
        if p.id in drop_ids:
            continue
        if p.parent_id is not None and p.parent_id in drop_ids:
            continue
        survivors.append(p)
    return drop_orphan_holes(survivors)

def drop_small_outer_polygons_scoped(
    polygons: list[PolygonData],
    min_area_px2: float,
    affected_ids: set[int],
) -> list[PolygonData]:
    if min_area_px2 <= 0.0 or not affected_ids:
        return polygons

    def _keep_outer(poly: PolygonData) -> bool:
        if poly.is_hole:
            return True
        if poly.category == "via" or poly.shape_hint == "box":
            return True
        return abs(float(poly.area)) >= float(min_area_px2)

    drop_ids = {
        polygon.id
        for polygon in polygons
        if polygon.id in affected_ids and not _keep_outer(polygon)
    }
    if not drop_ids:
        return polygons
    survivors: list[PolygonData] = []
    for polygon in polygons:
        if polygon.id in drop_ids:
            continue
        if polygon.parent_id is not None and polygon.parent_id in drop_ids:
            continue
        survivors.append(polygon)
    return drop_orphan_holes(survivors)

if TYPE_CHECKING:
    from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import VectorGeometrySettings
