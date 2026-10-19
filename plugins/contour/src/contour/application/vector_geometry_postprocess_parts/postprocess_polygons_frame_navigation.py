"""vector_geometry_postprocess: postprocess polygons frame navigation."""
from __future__ import annotations
from typing import TYPE_CHECKING
from math import hypot
from contour.domain import PolygonData, integer_point
from ...domain.polygon_ring import (
    is_valid_closed_polygon_edge_move,
)

from contour.application.vector_geometry_postprocess_parts.apply_spike_removal_polygon import (
    apply_spike_removal_all,
    clip_polygons_to_frame_raster,
)

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _refresh_metrics,
    collapse_redundant_vertices_in_polygons,
    dissolve_small_holes,
    drop_orphan_holes,
    drop_polygons_invalid_points,
    filter_simple_valid_polygons,
)

from contour.application.vector_geometry_postprocess_parts.drop_small_outer_polygons import drop_small_outer_polygons

from contour.application.vector_geometry_postprocess_parts.drop_triangle_outer_artifacts import drop_triangle_outer_artifacts

from contour.application.vector_geometry_postprocess_parts.repair_invalid_polygon_descriptions import _polygons_topo_signature

def postprocess_polygons_for_frame_navigation(
    polygons: list[PolygonData],
    frame_width: int,
    frame_height: int,
    settings: VectorGeometrySettings,
) -> tuple[list[PolygonData], bool]:
    """Called when syncing a newly opened frame."""

    before = _polygons_topo_signature(polygons)
    work = [p.clone() for p in polygons]
    work = drop_polygons_invalid_points(work)
    work = collapse_redundant_vertices_in_polygons(work)
    work = filter_simple_valid_polygons(work)
    if settings.clip_to_frame_on_sync and frame_width > 1 and frame_height > 1:
        work = clip_polygons_to_frame_raster(work, frame_width, frame_height)
    work = dissolve_small_holes(work, settings.min_hole_area_to_remove_px2)
    work = apply_spike_removal_all(work, settings.min_spike_interior_angle_deg)
    work = collapse_redundant_vertices_in_polygons(work)
    work = filter_simple_valid_polygons(work)
    work = drop_small_outer_polygons(work, settings.min_outer_area_px2)
    work = drop_triangle_outer_artifacts(
        work,
        settings.drop_three_vertex_triangle_artifacts,
        min_outer_area_px2=settings.min_outer_area_px2,
    )
    work = drop_polygons_invalid_points(work)
    work = filter_simple_valid_polygons(work)
    work = drop_orphan_holes(work)
    changed = before != _polygons_topo_signature(work)
    return work, changed

def _invalidate_authored_cif_paint_for_edit(
    polygons: list[PolygonData],
    target: PolygonData,
) -> None:
    """Discard authored family paint geometry after its recovered topology is edited."""

    by_id = {polygon.id: polygon for polygon in polygons}
    current: PolygonData | None = target
    visited: set[int] = set()
    while current is not None and current.id not in visited:
        visited.add(current.id)
        current.cif_paint_ring = []
        current = by_id.get(current.parent_id) if current.parent_id is not None else None

def apply_edge_translation_to_clone(
    polygons: list[PolygonData],
    polygon_id: int,
    edge_index: int,
    delta: tuple[float, float],
) -> list[PolygonData]:
    work = [p.clone() for p in polygons]
    by_id = {p.id: p for p in work}
    target = by_id.get(polygon_id)
    if target is None or edge_index < 0 or edge_index >= len(target.points):
        return work
    pts = [(float(x), float(y)) for x, y in target.points]
    dx, dy = float(delta[0]), float(delta[1])
    start_idx = edge_index
    end_idx = (edge_index + 1) % len(pts)
    closed_duplicate_endpoint = len(pts) > 2 and hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-5
    pts[start_idx] = integer_point((pts[start_idx][0] + dx, pts[start_idx][1] + dy))
    pts[end_idx] = integer_point((pts[end_idx][0] + dx, pts[end_idx][1] + dy))
    if closed_duplicate_endpoint and (start_idx == 0 or end_idx == 0 or start_idx == len(pts) - 1 or end_idx == len(pts) - 1):
        pts[-1] = pts[0]
    target.points = pts
    if not is_valid_closed_polygon_edge_move(target.points, edge_index):
        return [p.clone() for p in polygons]
    _invalidate_authored_cif_paint_for_edit(work, target)
    _refresh_metrics(target)
    return work

if TYPE_CHECKING:
    from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import VectorGeometrySettings
