"""vector_geometry_postprocess: resolve focus id after."""
from __future__ import annotations
from contour.domain import PolygonData, integer_points
from ...domain.polygon_ring import (
    is_valid_closed_polygon_ring,
)

from contour.application.vector_geometry_postprocess_parts.apply_spike_removal_polygon import _centroid

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _polygon_contains_point,
    _refresh_metrics,
)

from contour.application.vector_geometry_postprocess_parts.postprocess_polygons_frame_navigation import _invalidate_authored_cif_paint_for_edit

def apply_polygon_points_to_clone(
    polygons: list[PolygonData],
    polygon_id: int,
    new_points: list[tuple[float, float]],
) -> list[PolygonData]:
    work = [p.clone() for p in polygons]
    by_id = {p.id: p for p in work}
    target = by_id.get(polygon_id)
    if target is None:
        return work
    target.points = integer_points(new_points)
    if not is_valid_closed_polygon_ring(target.points):
        return [p.clone() for p in polygons]
    _invalidate_authored_cif_paint_for_edit(work, target)
    _refresh_metrics(target)
    return work

def resolve_focus_id_after_geometry_pass(
    before_polygons: list[PolygonData],
    polygon_id_hint: int,
    after_polygons: list[PolygonData],
) -> int | None:
    """Guess which polygon in ``after`` corresponds to the edited ``polygon_id_hint``."""

    before_by_id = {p.id: p for p in before_polygons}
    before_sel = before_by_id.get(polygon_id_hint)
    if before_sel is None:
        return after_polygons[0].id if after_polygons else None
    after_by_id = {polygon.id: polygon for polygon in after_polygons}
    if polygon_id_hint in after_by_id:
        return polygon_id_hint
    c = _centroid(before_sel.points)
    outer_candidates = [p for p in after_polygons if not p.is_hole]
    containment = []
    for p in outer_candidates:
        if _polygon_contains_point(p, c):
            containment.append(p)
    if not containment:
        return after_polygons[0].id if after_polygons else None
    best = min(containment, key=lambda p: abs(float(p.area)))
    return best.id
