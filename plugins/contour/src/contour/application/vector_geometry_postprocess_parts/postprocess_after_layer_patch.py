"""vector_geometry_postprocess: postprocess after layer patch."""
from __future__ import annotations
from contour.domain import PolygonData

from contour.application.vector_geometry_postprocess_parts.apply_spike_removal_polygon import (
    _affected_polygon_ids_for_layer_patch,
    apply_spike_removal_all,
    apply_spike_removal_for_polygon_ids,
    clip_polygons_to_frame_raster,
    drop_polygons_invalid_points_scoped,
    filter_simple_valid_polygons_scoped,
)

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _refresh_metrics,
    collapse_redundant_vertices_for_polygon_ids,
    collapse_redundant_vertices_in_polygons,
    dissolve_small_holes,
    dissolve_small_holes_scoped,
    drop_orphan_holes,
    drop_polygons_invalid_points,
    filter_simple_valid_polygons,
)

from contour.application.vector_geometry_postprocess_parts.drop_small_outer_polygons import (
    drop_small_outer_polygons,
    drop_small_outer_polygons_scoped,
)

from contour.application.vector_geometry_postprocess_parts.drop_triangle_outer_artifacts import (
    drop_triangle_outer_artifacts,
    drop_triangle_outer_artifacts_scoped,
)

from contour.application.vector_geometry_postprocess_parts.merge_overlapping_root_families import merge_overlapping_root_families

from contour.application.vector_geometry_postprocess_parts.repair_invalid_polygon_descriptions import _polygons_topo_signature

def postprocess_after_editor_mutation(
    polygons: list[PolygonData],
    settings: VectorGeometrySettings,
    *,
    frame_width_height: tuple[int, int] | None = None,
    include_merge: bool,
) -> tuple[list[PolygonData], bool]:
    """Apply cleanup after user edits.

    Pass ``frame_width_height`` only when callers want an extra clipping pass (normally editor relies on canvas).
    """

    before = _polygons_topo_signature(polygons)
    work = [p.clone() for p in polygons]
    work = drop_polygons_invalid_points(work)
    work = collapse_redundant_vertices_in_polygons(work)
    work = filter_simple_valid_polygons(work)
    for p in work:
        _refresh_metrics(p)

    fw, fh = frame_width_height or (0, 0)
    if settings.clip_to_frame_on_sync and fw > 1 and fh > 1:
        work = clip_polygons_to_frame_raster(work, fw, fh)

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

    if include_merge:
        work = merge_overlapping_root_families(work)

    work = drop_polygons_invalid_points(work)
    work = filter_simple_valid_polygons(work)

    work = drop_orphan_holes(work)
    changed = before != _polygons_topo_signature(work)
    return work, changed

def postprocess_after_layer_patch(
    polygons: list[PolygonData],
    settings: VectorGeometrySettings,
    *,
    changed_polygon_ids: set[int],
    frame_width_height: tuple[int, int] | None = None,
    include_merge: bool = False,
) -> tuple[list[PolygonData], bool]:
    """Apply cleanup after a local layer patch without rescanning the whole frame."""

    if not changed_polygon_ids:
        return postprocess_after_editor_mutation(
            polygons,
            settings,
            frame_width_height=frame_width_height,
            include_merge=include_merge,
        )

    work = list(polygons)
    affected_ids = _affected_polygon_ids_for_layer_patch(work, changed_polygon_ids)
    count_before = len(work)

    work = drop_polygons_invalid_points_scoped(work, affected_ids)
    work, _collapsed = collapse_redundant_vertices_for_polygon_ids(work, affected_ids)
    work = filter_simple_valid_polygons_scoped(work, affected_ids)
    for polygon in work:
        if polygon.id in affected_ids:
            _refresh_metrics(polygon)

    fw, fh = frame_width_height or (0, 0)
    if settings.clip_to_frame_on_sync and fw > 1 and fh > 1:
        work = clip_polygons_to_frame_raster(work, fw, fh)
        affected_ids = _affected_polygon_ids_for_layer_patch(work, changed_polygon_ids)

    work = dissolve_small_holes_scoped(work, settings.min_hole_area_to_remove_px2, affected_ids)
    work = apply_spike_removal_for_polygon_ids(
        work,
        settings.min_spike_interior_angle_deg,
        affected_ids,
    )
    work, _collapsed = collapse_redundant_vertices_for_polygon_ids(work, affected_ids)
    work = filter_simple_valid_polygons_scoped(work, affected_ids)

    work = drop_small_outer_polygons_scoped(work, settings.min_outer_area_px2, affected_ids)
    work = drop_triangle_outer_artifacts_scoped(
        work,
        affected_ids,
        settings.drop_three_vertex_triangle_artifacts,
        min_outer_area_px2=settings.min_outer_area_px2,
    )

    work = drop_polygons_invalid_points_scoped(work, affected_ids)
    work = filter_simple_valid_polygons_scoped(work, affected_ids)

    if include_merge:
        work = merge_overlapping_root_families(work)

    work = drop_polygons_invalid_points_scoped(work, affected_ids)
    work = filter_simple_valid_polygons_scoped(work, affected_ids)
    work = drop_orphan_holes(work)
    changed = len(work) != count_before or bool(affected_ids)
    return work, changed
