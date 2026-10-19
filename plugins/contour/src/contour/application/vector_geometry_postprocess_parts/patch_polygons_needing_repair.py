"""vector_geometry_postprocess: patch polygons needing repair."""
from __future__ import annotations
from typing import TYPE_CHECKING
from contour.domain import PolygonData

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _is_small_inner_area,
    _is_via_like,
)

from contour.application.vector_geometry_postprocess_parts.overlapping_root_family_ids import (
    apply_overlap_repair_patch,
    overlap_check_roots_for_layer_patch,
)

def patch_polygons_needing_repair(
    existing: dict[int, list[str]],
    polygons: list[PolygonData],
    *,
    removed_ids: set[int],
    removed_polygons: list[PolygonData],
    added_or_changed: list[PolygonData],
    settings: VectorGeometrySettings,
    include_overlap: bool = True,
) -> dict[int, list[str]]:
    """Update a repair map after an incremental polygon-layer edit."""

    reasons_by_id = {
        polygon_id: list(reason_codes)
        for polygon_id, reason_codes in existing.items()
        if polygon_id not in removed_ids
    }

    def _add(polygon_id: int, reason: str) -> None:
        existing_reasons = reasons_by_id.setdefault(polygon_id, [])
        if reason not in existing_reasons:
            existing_reasons.append(reason)

    min_outer = float(settings.min_outer_area_px2)
    min_hole = float(settings.min_hole_area_to_remove_px2)
    for polygon in added_or_changed:
        if polygon.description_is_invalid():
            ring_reason = polygon.description_invalid_reason()
            if ring_reason is not None:
                _add(polygon.id, ring_reason)
        if polygon.is_hole:
            if min_hole > 0.0 and _is_small_inner_area(polygon, min_hole):
                _add(polygon.id, "small_hole")
            continue
        if _is_via_like(polygon):
            continue
        if min_outer > 0.0 and abs(float(polygon.area)) < min_outer:
            _add(polygon.id, "small_object")

    if not include_overlap:
        return reasons_by_id

    overlap_check_roots = overlap_check_roots_for_layer_patch(
        polygons,
        removed_polygons,
        added_or_changed,
    )
    return apply_overlap_repair_patch(reasons_by_id, polygons, overlap_check_roots)

if TYPE_CHECKING:
    from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import VectorGeometrySettings
