"""vector_geometry_postprocess: postprocess vertex move edit."""
from __future__ import annotations
from math import hypot
from contour.domain import PolygonData

from contour.application.vector_geometry_postprocess_parts.apply_vertex_position_scoped import apply_vertex_position_scoped

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _refresh_metrics,
    collapse_redundant_vertices_for_polygon_ids,
    drop_orphan_holes,
    drop_polygons_invalid_points,
    filter_simple_valid_polygons,
)

from contour.application.vector_geometry_postprocess_parts.merge_overlapping_root_families import merge_overlapping_root_families_near_polygons

from contour.application.vector_geometry_postprocess_parts.postprocess_changed_polygon_edit import postprocess_changed_polygon_edit

from contour.application.vector_geometry_postprocess_parts.repair_invalid_polygon_descriptions import _polygons_topo_signature

from contour.application.vector_geometry_postprocess_parts.resolve_focus_id_after import resolve_focus_id_after_geometry_pass

def postprocess_vertex_move_edit(
    polygons: list[PolygonData],
    settings: VectorGeometrySettings,
    *,
    polygon_id: int,
    vertex_index: int,
    new_point: tuple[float, float],
) -> tuple[list[PolygonData], bool, int | None]:
    """Fast vertex-move postprocess with structural sharing and scoped collapse."""

    source = next((polygon for polygon in polygons if polygon.id == polygon_id), None)
    preserve_duplicate_closure = bool(
        source is not None
        and len(source.points) > 3
        and hypot(
            float(source.points[0][0]) - float(source.points[-1][0]),
            float(source.points[0][1]) - float(source.points[-1][1]),
        )
        < 1e-5
    )

    work, moved = apply_vertex_position_scoped(
        polygons,
        polygon_id,
        vertex_index,
        new_point,
    )
    if not moved:
        return list(polygons), False, polygon_id

    work, accepted, _changed = postprocess_changed_polygon_edit(
        work,
        settings,
        polygon_id=polygon_id,
    )
    if not accepted:
        return list(polygons), False, polygon_id

    if polygon_id in {polygon.id for polygon in work}:
        work, _collapsed_ids = collapse_redundant_vertices_for_polygon_ids(work, {polygon_id})
        if preserve_duplicate_closure:
            moved_polygon = next(polygon for polygon in work if polygon.id == polygon_id)
            if moved_polygon.points and moved_polygon.points[-1] != moved_polygon.points[0]:
                moved_polygon.points = [*moved_polygon.points, moved_polygon.points[0]]
                _refresh_metrics(moved_polygon)
        if settings.merge_overlapping_on_edit:
            before_merge = _polygons_topo_signature(work)
            work = merge_overlapping_root_families_near_polygons(
                work,
                polygon_id,
                expand_touching_roots=False,
            )
            work = drop_polygons_invalid_points(work)
            work = filter_simple_valid_polygons(work)
            work = drop_orphan_holes(work)
            if before_merge != _polygons_topo_signature(work) and polygon_id not in {polygon.id for polygon in work}:
                return work, True, resolve_focus_id_after_geometry_pass(polygons, polygon_id, work)
        return work, True, polygon_id

    work = drop_orphan_holes(work)
    return work, True, resolve_focus_id_after_geometry_pass(polygons, polygon_id, work)
