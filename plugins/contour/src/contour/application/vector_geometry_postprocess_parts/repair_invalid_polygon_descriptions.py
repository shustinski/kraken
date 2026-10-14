"""vector_geometry_postprocess: repair invalid polygon descriptions."""
from __future__ import annotations
from contour.domain import PolygonData
from ...domain.polygon_ring import (
    TOPOLOGY_CHECK_MAX_VERTICES,
)

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    collapse_redundant_vertices_in_polygons,
    dissolve_small_holes,
    drop_orphan_holes,
)

from contour.application.vector_geometry_postprocess_parts.dissolve_self_intersecting_polygons import dissolve_self_intersecting_polygons

from contour.application.vector_geometry_postprocess_parts.drop_small_outer_polygons import drop_small_outer_polygons

from contour.application.vector_geometry_postprocess_parts.merge_overlapping_root_families import merge_overlapping_root_families

from contour.application.vector_geometry_postprocess_parts.polygons_needing_repair import polygons_needing_repair

from contour.application.vector_geometry_postprocess_parts.rebuild_family_filled_mask import (
    _polygon_ring_has_keyhole_bridge,
    _polygon_ring_needs_explicit_topology_repair,
)

def summarize_invalid_polygon_description_reasons(
    polygons: list[PolygonData],
    settings: VectorGeometrySettings | None = None,
) -> list[tuple[str, int]]:
    """Count repair reason codes across polygons, preserving first-seen order.

    When ``settings`` is omitted, only ring-local invalid-description reasons are counted
    (backwards-compatible with topology-only callers). With settings, also counts
    overlapping / small-object / small-hole reasons using the same thresholds as manual postprocess.
    """

    counts: dict[str, int] = {}
    order: list[str] = []

    def _bump(reason: str) -> None:
        if reason not in counts:
            order.append(reason)
            counts[reason] = 0
        counts[reason] += 1

    if settings is None:
        for polygon in polygons:
            reason = polygon.description_invalid_reason()
            if reason is not None:
                _bump(reason)
    else:
        reasons_by_id = polygons_needing_repair(polygons, settings)
        for polygon in polygons:
            for reason in reasons_by_id.get(polygon.id, ()):
                _bump(reason)
    return [(reason, counts[reason]) for reason in order]

def repair_invalid_polygon_descriptions(
    polygons: list[PolygonData],
    settings: VectorGeometrySettings | None = None,
) -> list[PolygonData]:
    """Repair rings that need cleanup.

    Without ``settings``: rebuild keyholes / self-intersections only (legacy topology path).
    With ``settings``: also merge overlapping root families, dissolve small holes, and drop
    small outers using the same thresholds as manual vector postprocess.

    CIF keyhole ``repeated_vertex`` rings are not auto-offered as invalid, but explicit
    repair still splits them when this function is invoked.
    """

    if settings is None:
        if not any(_polygon_ring_needs_explicit_topology_repair(polygon) for polygon in polygons):
            return [polygon.clone() for polygon in polygons]
        return dissolve_self_intersecting_polygons(polygons, include_keyhole_bridges=True)

    needs = polygons_needing_repair(polygons, settings)
    has_keyhole = any(_polygon_ring_has_keyhole_bridge(polygon) for polygon in polygons)
    if not needs and not has_keyhole:
        return [polygon.clone() for polygon in polygons]

    all_reasons = {reason for reasons in needs.values() for reason in reasons}
    work = [polygon.clone() for polygon in polygons]

    if has_keyhole or any(reason in {"repeated_vertex", "self_intersecting"} for reason in all_reasons):
        work = dissolve_self_intersecting_polygons(work, include_keyhole_bridges=True)

    if "overlapping" in all_reasons:
        work = merge_overlapping_root_families(work)
        # Raster merge of adjacent healed rings can recreate keyhole descriptions; heal again.
        work = dissolve_self_intersecting_polygons(work, include_keyhole_bridges=True)

    work = dissolve_small_holes(work, settings.min_hole_area_to_remove_px2)
    work = drop_small_outer_polygons(work, settings.min_outer_area_px2)
    work = drop_orphan_holes(work)
    return collapse_redundant_vertices_in_polygons(work)

def _polygon_topo_points_key(points: list[tuple[float, float]]) -> object:
    n = len(points)
    if n <= TOPOLOGY_CHECK_MAX_VERTICES:
        return tuple((round(float(x_coord), 4), round(float(y_coord), 4)) for x_coord, y_coord in points)
    centroid_x = 0.0
    centroid_y = 0.0
    for x_coord, y_coord in points:
        centroid_x += float(x_coord)
        centroid_y += float(y_coord)
    inv_n = 1.0 / float(n)
    return (
        "dense",
        n,
        round(centroid_x * inv_n * 10_000.0),
        round(centroid_y * inv_n * 10_000.0),
    )

def _single_polygon_topo_signature(polygon: PolygonData) -> tuple[object, ...]:
    return (
        polygon.id,
        bool(polygon.is_hole),
        polygon.parent_id,
        str(polygon.category),
        str(polygon.shape_hint),
        _polygon_topo_points_key(polygon.points),
    )

def _polygons_topo_signature(polygons: list[PolygonData]) -> tuple[tuple[object, ...], ...]:
    return tuple(_single_polygon_topo_signature(polygon) for polygon in sorted(polygons, key=lambda q: q.id))
