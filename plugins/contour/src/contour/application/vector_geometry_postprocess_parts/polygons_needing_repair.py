"""vector_geometry_postprocess: polygons needing repair."""
from __future__ import annotations
from typing import TYPE_CHECKING
from contour.domain import PolygonData

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _is_small_inner_area,
    _is_via_like,
)

from contour.application.vector_geometry_postprocess_parts.overlapping_root_family_ids import overlapping_root_family_ids

from contour.application.vector_geometry_postprocess_parts.root_id_map import _collect_family

def polygons_needing_repair(
    polygons: list[PolygonData],
    settings: VectorGeometrySettings,
    *,
    include_ring_reasons: bool = True,
    include_overlap: bool = True,
) -> dict[int, list[str]]:
    """Map polygon id → ordered unique repair reason codes (topology + area + overlap).

    Reason codes:
    - ``self_intersecting`` — ring-local invalid description (bowtie / crossing)
    - ``overlapping`` — filled root family intersects another conductor root
    - ``small_object`` — non-via outer below ``settings.min_outer_area_px2``
    - ``small_hole`` — hole below ``settings.min_hole_area_to_remove_px2``

    Note: CIF keyhole ``repeated_vertex`` is intentionally omitted here so auto red-mark
    and repair offers do not treat valid keyhole ``P`` rings as broken.
    """

    reasons_by_id: dict[int, list[str]] = {}

    def _add(polygon_id: int, reason: str) -> None:
        existing = reasons_by_id.setdefault(polygon_id, [])
        if reason not in existing:
            existing.append(reason)

    if include_ring_reasons:
        for polygon in polygons:
            if polygon.description_is_invalid():
                ring_reason = polygon.description_invalid_reason()
                if ring_reason is not None:
                    _add(polygon.id, ring_reason)

    if not include_overlap:
        min_outer = float(settings.min_outer_area_px2)
        min_hole = float(settings.min_hole_area_to_remove_px2)
        for polygon in polygons:
            if polygon.is_hole:
                if min_hole > 0.0 and _is_small_inner_area(polygon, min_hole):
                    _add(polygon.id, "small_hole")
                continue
            if _is_via_like(polygon):
                continue
            if min_outer > 0.0 and abs(float(polygon.area)) < min_outer:
                _add(polygon.id, "small_object")
        return reasons_by_id

    overlapping_roots = overlapping_root_family_ids(polygons)
    for root_id in overlapping_roots:
        for member in _collect_family(polygons, root_id):
            _add(member.id, "overlapping")

    min_outer = float(settings.min_outer_area_px2)
    min_hole = float(settings.min_hole_area_to_remove_px2)
    for polygon in polygons:
        if polygon.is_hole:
            if min_hole > 0.0 and _is_small_inner_area(polygon, min_hole):
                _add(polygon.id, "small_hole")
            continue
        if _is_via_like(polygon):
            continue
        if min_outer > 0.0 and abs(float(polygon.area)) < min_outer:
            _add(polygon.id, "small_object")

    return reasons_by_id

if TYPE_CHECKING:
    from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import VectorGeometrySettings
