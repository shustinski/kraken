"""vector_geometry_postprocess: apply vertex position scoped."""
from __future__ import annotations
from math import hypot
from contour.domain import PolygonData, integer_point, integer_points
from ...domain.polygon_ring import (
    is_valid_closed_polygon_vertex_move,
)

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _refresh_metrics,
    _replace_polygon_in_list,
)

from contour.application.vector_geometry_postprocess_parts.postprocess_polygons_frame_navigation import _invalidate_authored_cif_paint_for_edit

def apply_vertex_position_to_clone(
    polygons: list[PolygonData],
    polygon_id: int,
    vertex_index: int,
    new_point: tuple[float, float],
) -> list[PolygonData]:
    work, moved = apply_vertex_position_scoped(polygons, polygon_id, vertex_index, new_point)
    if moved:
        return work
    return [polygon.clone() for polygon in polygons]

def apply_vertex_position_scoped(
    polygons: list[PolygonData],
    polygon_id: int,
    vertex_index: int,
    new_point: tuple[float, float],
) -> tuple[list[PolygonData], bool]:
    """Apply a vertex move by cloning only the edited polygon."""

    target = next((polygon for polygon in polygons if polygon.id == polygon_id), None)
    if target is None or vertex_index < 0 or vertex_index >= len(target.points):
        return list(polygons), False
    clone = target.clone()
    pts = [(float(x), float(y)) for x, y in clone.points]
    moved_point = integer_point(new_point)
    closed_duplicate_endpoint = len(pts) > 2 and hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-5
    pts[vertex_index] = moved_point
    if closed_duplicate_endpoint:
        if vertex_index == 0:
            pts[-1] = moved_point
        elif vertex_index == len(pts) - 1:
            pts[0] = moved_point
    clone.points = pts
    if not is_valid_closed_polygon_vertex_move(clone.points, vertex_index):
        return list(polygons), False
    work = _replace_polygon_in_list(polygons, clone)
    _invalidate_authored_cif_paint_for_edit(work, clone)
    _refresh_metrics(clone)
    return work, True

def apply_vertex_delete_to_clone(
    polygons: list[PolygonData],
    polygon_id: int,
    vertex_index: int,
) -> list[PolygonData]:
    work = [polygon.clone() for polygon in polygons]
    target = next((polygon for polygon in work if polygon.id == polygon_id), None)
    if target is None or vertex_index < 0 or vertex_index >= len(target.points):
        return work
    if len(target.points) <= 3:
        return work
    points = [(float(x_coord), float(y_coord)) for x_coord, y_coord in target.points]
    points.pop(vertex_index)
    target.points = integer_points(points)
    _invalidate_authored_cif_paint_for_edit(work, target)
    _refresh_metrics(target)
    return work
