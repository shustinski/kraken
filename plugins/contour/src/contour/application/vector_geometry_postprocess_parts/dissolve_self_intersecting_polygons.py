"""vector_geometry_postprocess: dissolve self intersecting polygons."""
from __future__ import annotations
from contour.domain import PolygonData

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _refresh_metrics,
    _root_id_for_polygon,
    drop_orphan_holes,
)

from contour.application.vector_geometry_postprocess_parts.overlapping_root_family_ids import _dissolve_family_to_simple_rings

from contour.application.vector_geometry_postprocess_parts.rebuild_family_filled_mask import (
    _polygon_ring_needs_dissolve,
    _polygon_ring_needs_explicit_topology_repair,
)

from contour.application.vector_geometry_postprocess_parts.root_id_map import (
    _collect_family,
    _next_polygon_id_after,
    _relabel_polygon_ids,
    _stamp_visual_metadata,
)

def dissolve_self_intersecting_polygons(
    polygons: list[PolygonData],
    *,
    include_keyhole_bridges: bool = False,
    changed_polygon_ids: set[int] | None = None,
) -> list[PolygonData]:
    """Rebuild rings whose new edges cross existing ones into a simple filled union.

    By default only true invalids (e.g. bowties) are rewritten. Pass
    ``include_keyhole_bridges=True`` for explicit repair that also splits CIF
    keyhole ``repeated_vertex`` bridges into outer+hole children.

    When ``changed_polygon_ids`` is set, only scan and rebuild root families
    touched by those edits (unchanged rings keep their cached validity).
    """

    def _needs(polygon: PolygonData) -> bool:
        if include_keyhole_bridges:
            return _polygon_ring_needs_explicit_topology_repair(polygon)
        return _polygon_ring_needs_dissolve(polygon)

    roots_to_process: set[int] | None = None
    if changed_polygon_ids is not None:
        if not changed_polygon_ids:
            return polygons
        polygons_by_id = {polygon.id: polygon for polygon in polygons}
        roots_to_process = {
            root_id
            for polygon_id in changed_polygon_ids
            if (root_id := _root_id_for_polygon(polygons_by_id, polygon_id)) is not None
        }
        if not roots_to_process:
            return polygons
        if not any(
            _needs(polygons_by_id[polygon_id])
            for polygon_id in changed_polygon_ids
            if polygon_id in polygons_by_id
        ):
            return polygons
    elif not any(_needs(polygon) for polygon in polygons):
        return polygons

    roots = [polygon.id for polygon in polygons if polygon.parent_id is None]
    consumed_ids: set[int] = set()
    rebuilt: list[PolygonData] = []
    next_id = _next_polygon_id_after(polygons)
    for root_id in roots:
        if roots_to_process is not None and root_id not in roots_to_process:
            continue
        family = _collect_family(polygons, root_id)
        if changed_polygon_ids is not None:
            family_changed = [
                polygon for polygon in family if polygon.id in changed_polygon_ids
            ]
            if not family_changed or not any(_needs(polygon) for polygon in family_changed):
                continue
        elif not any(_needs(polygon) for polygon in family):
            continue
        extracted = _dissolve_family_to_simple_rings(family)
        if not extracted:
            continue
        _stamp_visual_metadata(family, extracted)
        for polygon in extracted:
            _refresh_metrics(polygon)
        extracted = _relabel_polygon_ids(extracted, start_id=next_id)
        next_id = _next_polygon_id_after(extracted)
        rebuilt.extend(extracted)
        consumed_ids.update(polygon.id for polygon in family)

    if not consumed_ids:
        return polygons
    leftovers = [polygon.clone() for polygon in polygons if polygon.id not in consumed_ids]
    return drop_orphan_holes(leftovers + rebuilt)
