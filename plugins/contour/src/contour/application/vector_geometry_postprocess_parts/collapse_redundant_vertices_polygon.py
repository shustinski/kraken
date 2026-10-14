"""vector_geometry_postprocess: collapse redundant vertices polygon."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from contour.domain import PolygonData, compute_polygon_metrics
from ...domain.polygon_ring import (
    TOPOLOGY_CHECK_MAX_VERTICES,
    collapse_redundant_polyline_vertices,
    is_valid_closed_polygon_ring,
)

def _ring_ok_after_spike_removal(points: list[tuple[float, float]]) -> bool:
    if len(points) < 3:
        return False
    if len(points) > TOPOLOGY_CHECK_MAX_VERTICES:
        return True
    return is_valid_closed_polygon_ring(points)

def _polygon_contains_point(poly: PolygonData, point: tuple[float, float]) -> bool:
    from ...graphics.geometry import _polygon_contains_point as impl

    return impl(poly, point)

def _mask_helpers():
    from contour.application.vector_geometry_postprocess_parts import module_state

    if module_state._MASK_HELPERS is None:
        from ...graphics.geometry import (
            _bbox_from_points,
            _polygons_from_mask,
            _render_polygon_collection_on_mask,
            _union_bbox,
        )

        module_state._MASK_HELPERS = (
            _bbox_from_points,
            _polygons_from_mask,
            _render_polygon_collection_on_mask,
            _union_bbox,
        )
    return module_state._MASK_HELPERS

def _bboxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by

@dataclass(slots=True)
class VectorGeometrySettings:
    """Editor / frame-sync vector cleanup (independent of mask extraction settings)."""

    clip_to_frame_on_sync: bool = True
    #: Minimum oriented area (px²) for an outer ring; excludes vias (`category=="via"` or `shape_hint=="box"`).
    min_outer_area_px2: float = 60_000.0
    #: Holes smaller than this area are filled; ``0`` disables.
    min_hole_area_to_remove_px2: float = 100_000.0
    #: Merge conductors whose filled regions intersect after edits (vertex / polygon moves).
    merge_overlapping_on_edit: bool = True
    #: Local edits touching at least this many polygons may defer overlap merge after commit.
    #: Topology repair (self-intersection dissolve) always runs synchronously.
    deferred_geometry_repair_affected_threshold: int = 48
    #: Interior angle threshold in degrees; spikes with a smaller apex angle at a vertex are removed. ``0`` disables.
    min_spike_interior_angle_deg: float = 30.0
    #: Drop unparented 3-vertex **outer** polygons (non-via) as triangle artifacts — disable if intentional triangles occur.
    drop_three_vertex_triangle_artifacts: bool = True

def _point_finite(p: tuple[float, float]) -> bool:
    return np.isfinite(p[0]) and np.isfinite(p[1])

def _refresh_metrics(poly: PolygonData) -> None:
    area, perimeter, bbox = compute_polygon_metrics(poly.points)
    poly.area = float(area)
    poly.perimeter = float(perimeter)
    poly.bbox = bbox

def _ring_has_usable_geometry(points: list[tuple[float, float]]) -> bool:
    from ...graphics.brush_vector import ring_is_valid_for_polygon

    return ring_is_valid_for_polygon(points)

def collapse_redundant_vertices_in_polygons(polygons: list[PolygonData]) -> list[PolygonData]:
    collapsed: list[PolygonData] = []
    for polygon in polygons:
        clone = polygon.clone()
        new_points = collapse_redundant_polyline_vertices(clone.points, closed=True, min_vertices=3)
        if len(new_points) >= 3:
            clone.points = new_points
            _refresh_metrics(clone)
        collapsed.append(clone)
    return collapsed

def collapse_redundant_vertices_for_polygon_ids(
    polygons: list[PolygonData],
    polygon_ids: set[int],
) -> tuple[list[PolygonData], set[int]]:
    """Collapse rings for ``polygon_ids`` only; reuse untouched polygon references."""

    if not polygon_ids:
        return polygons, set()
    index_by_id = {polygon.id: index for index, polygon in enumerate(polygons)}
    work = list(polygons)
    changed_ids: set[int] = set()
    for polygon_id in polygon_ids:
        index = index_by_id.get(polygon_id)
        if index is None:
            continue
        polygon = work[index]
        new_points = collapse_redundant_polyline_vertices(polygon.points, closed=True, min_vertices=3)
        if len(new_points) < 3 or new_points == polygon.points:
            continue
        clone = polygon.clone()
        clone.points = new_points
        _refresh_metrics(clone)
        work[index] = clone
        changed_ids.add(polygon_id)
    return work, changed_ids

def _replace_polygon_in_list(polygons: list[PolygonData], replacement: PolygonData) -> list[PolygonData]:
    work = list(polygons)
    for index, polygon in enumerate(work):
        if polygon.id == replacement.id:
            work[index] = replacement
            return work
    return work

def drop_polygons_invalid_points(polygons: list[PolygonData]) -> list[PolygonData]:
    keep: list[PolygonData] = []
    for p in polygons:
        if not _ring_has_usable_geometry(p.points):
            continue
        if any(not _point_finite(pt) for pt in p.points):
            continue
        keep.append(p)
    return keep

def filter_simple_valid_polygons(polygons: list[PolygonData]) -> list[PolygonData]:
    # Keep mask-extracted rings even when :func:`is_valid_closed_polygon_ring` rejects them,
    # otherwise hierarchy (outer + holes) silently collapses during editor post-process.
    return [p for p in polygons if _ring_has_usable_geometry(p.points)]

def _is_small_inner_area(poly: PolygonData, min_area_px2: float) -> bool:
    if min_area_px2 <= 0.0:
        return False
    area = abs(float(poly.area))
    return area < float(min_area_px2)

def dissolve_small_holes(polygons: list[PolygonData], min_area_px2: float) -> list[PolygonData]:
    if min_area_px2 <= 0.0:
        return polygons
    return [
        polygon
        for polygon in polygons
        if not (polygon.is_hole and _is_small_inner_area(polygon, min_area_px2))
    ]

def dissolve_small_holes_scoped(
    polygons: list[PolygonData],
    min_area_px2: float,
    affected_ids: set[int],
) -> list[PolygonData]:
    if min_area_px2 <= 0.0 or not affected_ids:
        return polygons
    return [
        polygon
        for polygon in polygons
        if not (
            polygon.id in affected_ids
            and polygon.is_hole
            and _is_small_inner_area(polygon, min_area_px2)
        )
    ]

def drop_orphan_holes(polygons: list[PolygonData]) -> list[PolygonData]:
    ids = {p.id for p in polygons}
    return [p for p in polygons if p.parent_id is None or p.parent_id in ids]

def _is_via_like(polygon: PolygonData) -> bool:
    return str(polygon.category) == "via" or str(polygon.shape_hint) == "box"

def reparent_polygons_orphaned_by_ids(
    polygons: list[PolygonData],
    removed_ids: set[int],
) -> list[PolygonData]:
    """Keep islands that lived inside a deleted hole as independent filled roots."""

    if not removed_ids:
        return [polygon.clone() for polygon in polygons]
    reparented: list[PolygonData] = []
    for polygon in polygons:
        clone = polygon.clone()
        if clone.parent_id in removed_ids:
            clone.parent_id = None
            if not _is_via_like(clone):
                clone.is_hole = False
        reparented.append(clone)
    return reparented

def _root_id_for_polygon(polygons_by_id: dict[int, PolygonData], polygon_id: int) -> int | None:
    current = polygon_id
    seen: set[int] = set()
    while current in polygons_by_id:
        if current in seen:
            return current
        seen.add(current)
        polygon = polygons_by_id[current]
        if polygon.parent_id is None:
            return current
        current = polygon.parent_id
    return None
