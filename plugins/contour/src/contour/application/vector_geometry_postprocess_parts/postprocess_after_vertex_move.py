"""vector_geometry_postprocess: postprocess after vertex move."""
from __future__ import annotations
from contour.domain import PolygonData

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    drop_orphan_holes,
    drop_polygons_invalid_points,
    filter_simple_valid_polygons,
)

from contour.application.vector_geometry_postprocess_parts.merge_overlapping_root_families import merge_overlapping_root_families_near_polygons

from contour.application.vector_geometry_postprocess_parts.postprocess_changed_polygon_edit import postprocess_changed_polygon_edit

from contour.application.vector_geometry_postprocess_parts.repair_invalid_polygon_descriptions import _polygons_topo_signature

def postprocess_after_vertex_move(
    polygons: list[PolygonData],
    settings: VectorGeometrySettings,
    *,
    polygon_id: int | None,
) -> tuple[list[PolygonData], bool]:
    """Vertex-move postprocess: local cleanup on the edited polygon, then optional family merge."""

    work, accepted, changed = postprocess_changed_polygon_edit(polygons, settings, polygon_id=polygon_id)
    if not accepted:
        return [p.clone() for p in polygons], False
    if settings.merge_overlapping_on_edit and polygon_id is not None:
        before_merge = _polygons_topo_signature(work)
        work = merge_overlapping_root_families_near_polygons(
            work,
            polygon_id,
            expand_touching_roots=False,
        )
        work = drop_polygons_invalid_points(work)
        work = filter_simple_valid_polygons(work)
        work = drop_orphan_holes(work)
        if before_merge != _polygons_topo_signature(work):
            changed = True
    return work, changed
