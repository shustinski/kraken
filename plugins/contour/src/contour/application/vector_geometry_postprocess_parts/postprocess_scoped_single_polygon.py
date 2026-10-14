"""vector_geometry_postprocess: postprocess scoped single polygon."""
from __future__ import annotations
from contour.domain import PolygonData

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _point_finite,
    _refresh_metrics,
    dissolve_small_holes,
    filter_simple_valid_polygons,
)

from contour.application.vector_geometry_postprocess_parts.drop_small_outer_polygons import drop_small_outer_polygons

from contour.application.vector_geometry_postprocess_parts.drop_triangle_outer_artifacts import drop_triangle_outer_artifacts

def _postprocess_scoped_single_polygon(
    target: PolygonData,
    settings: VectorGeometrySettings,
) -> PolygonData | None:
    if len(target.points) < 3 or any(not _point_finite(pt) for pt in target.points):
        return None

    _refresh_metrics(target)
    scoped = [target.clone()]
    if target.is_hole:
        scoped = filter_simple_valid_polygons(scoped)
    else:
        scoped = dissolve_small_holes(scoped, settings.min_hole_area_to_remove_px2)
        # A direct edit is transactional: keep any valid geometry the user
        # authored. Automatic spike simplification belongs to an explicit/full
        # cleanup pass and must not silently turn a vertex drag into a rollback.
        scoped = filter_simple_valid_polygons(scoped)
        scoped = drop_small_outer_polygons(scoped, settings.min_outer_area_px2)
        scoped = drop_triangle_outer_artifacts(
            scoped,
            settings.drop_three_vertex_triangle_artifacts,
            min_outer_area_px2=settings.min_outer_area_px2,
        )
    if not scoped:
        return None
    return scoped[0]
