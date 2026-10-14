"""vector_geometry_postprocess: apply spike removal polygon."""
from __future__ import annotations
from math import atan2, hypot, pi, radians
import numpy as np
from contour.domain import PolygonData, integer_points
from ...domain.polygon_ring import (
    TOPOLOGY_CHECK_MAX_VERTICES,
)

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _mask_helpers,
    _point_finite,
    _refresh_metrics,
    _ring_has_usable_geometry,
    _ring_ok_after_spike_removal,
    _root_id_for_polygon,
    drop_orphan_holes,
)

def _interior_turn_angle_rad(prev_b: tuple[float, float], b: tuple[float, float], next_b: tuple[float, float]) -> float:
    """Smaller sweep angle between edges (prev→b) and (next→b) meeting at ``b``; in ``(0, π]``."""

    ux, uy = prev_b[0] - b[0], prev_b[1] - b[1]
    vx, vy = next_b[0] - b[0], next_b[1] - b[1]
    nu = hypot(ux, uy)
    nv = hypot(vx, vy)
    if nu < 1e-12 or nv < 1e-12:
        return pi
    cross = ux * vy - uy * vx
    dot = ux * vx + uy * vy
    return abs(atan2(cross, dot))

def remove_spikes_from_polygon_ring(points: list[tuple[float, float]], min_interior_angle_deg: float) -> list[tuple[float, float]]:
    if min_interior_angle_deg <= 0.0 or len(points) < 4:
        return points
    min_rad = radians(min_interior_angle_deg)
    pts = list(points)
    safety_cap = max(256, min(len(points) * 8, len(points) * len(points)))
    iterations = 0
    changed = True
    while changed and len(pts) >= 4 and iterations < safety_cap:
        iterations += 1
        changed = False
        n = len(pts)
        kill: int | None = None
        for i in range(n):
            prev_p = pts[(i - 1) % n]
            curr = pts[i]
            next_p = pts[(i + 1) % n]
            ang = _interior_turn_angle_rad(prev_p, curr, next_p)
            if ang + 1e-9 < min_rad:
                kill = i
                changed = True
                break
        if kill is not None:
            pts.pop(kill)
    return pts

def apply_spike_removal_all(polygons: list[PolygonData], min_interior_angle_deg: float) -> list[PolygonData]:
    if min_interior_angle_deg <= 0.0:
        return polygons
    out: list[PolygonData] = []
    for p in polygons:
        if p.is_hole:
            out.append(p.clone())
            continue
        if len(p.points) > TOPOLOGY_CHECK_MAX_VERTICES:
            out.append(p.clone())
            continue
        q = p.clone()
        new_pts = remove_spikes_from_polygon_ring(integer_points(q.points), min_interior_angle_deg)
        if len(new_pts) < 3 or not _ring_ok_after_spike_removal(new_pts):
            out.append(p.clone())
            continue
        q.points = new_pts
        _refresh_metrics(q)
        out.append(q)
    out = drop_orphan_holes(out)
    return out

def apply_spike_removal_for_polygon_ids(
    polygons: list[PolygonData],
    min_interior_angle_deg: float,
    polygon_ids: set[int],
) -> list[PolygonData]:
    if min_interior_angle_deg <= 0.0 or not polygon_ids:
        return polygons
    work = list(polygons)
    index_by_id = {polygon.id: index for index, polygon in enumerate(work)}
    for polygon_id in polygon_ids:
        index = index_by_id.get(polygon_id)
        if index is None:
            continue
        polygon = work[index]
        if polygon.is_hole or len(polygon.points) > TOPOLOGY_CHECK_MAX_VERTICES:
            continue
        clone = polygon.clone()
        new_pts = remove_spikes_from_polygon_ring(
            integer_points(clone.points),
            min_interior_angle_deg,
        )
        if len(new_pts) < 3 or not _ring_ok_after_spike_removal(new_pts):
            continue
        clone.points = new_pts
        _refresh_metrics(clone)
        work[index] = clone
    return drop_orphan_holes(work)

def drop_polygons_invalid_points_scoped(
    polygons: list[PolygonData],
    affected_ids: set[int],
) -> list[PolygonData]:
    if not affected_ids:
        return polygons
    kept: list[PolygonData] = []
    for polygon in polygons:
        if polygon.id not in affected_ids:
            kept.append(polygon)
            continue
        if not _ring_has_usable_geometry(polygon.points):
            continue
        if any(not _point_finite(point) for point in polygon.points):
            continue
        kept.append(polygon)
    return kept

def filter_simple_valid_polygons_scoped(
    polygons: list[PolygonData],
    affected_ids: set[int],
) -> list[PolygonData]:
    if not affected_ids:
        return polygons
    kept: list[PolygonData] = []
    for polygon in polygons:
        if polygon.id not in affected_ids or _ring_has_usable_geometry(polygon.points):
            kept.append(polygon)
    return kept

def _polygon_ids_in_root_families(polygons: list[PolygonData], root_ids: set[int]) -> set[int]:
    if not root_ids:
        return set()
    by_id = {polygon.id: polygon for polygon in polygons}
    affected: set[int] = set()
    for polygon in polygons:
        root_id = _root_id_for_polygon(by_id, polygon.id)
        if root_id in root_ids:
            affected.add(polygon.id)
    return affected

def _affected_polygon_ids_for_layer_patch(
    polygons: list[PolygonData],
    changed_polygon_ids: set[int],
) -> set[int]:
    root_ids = {
        root_id
        for root_id in (
            _root_id_for_polygon({polygon.id: polygon for polygon in polygons}, polygon_id)
            for polygon_id in changed_polygon_ids
        )
        if root_id is not None
    }
    return _polygon_ids_in_root_families(polygons, root_ids)

def clip_polygons_to_frame_raster(polygons: list[PolygonData], frame_width: int, frame_height: int) -> list[PolygonData]:
    from contour.application.vector_geometry_postprocess_parts.root_id_map import _stamp_visual_metadata
    _, _polygons_from_mask, _render_polygon_collection_on_mask, _ = _mask_helpers()
    if frame_width <= 1 or frame_height <= 1 or not polygons:
        return polygons
    width = max(1, int(frame_width))
    height = max(1, int(frame_height))
    mask = np.zeros((height, width), dtype=np.uint8)
    origin = (0, 0)
    try:
        _render_polygon_collection_on_mask(mask, [p.clone() for p in polygons], origin)
    except Exception:
        return polygons
    extracted = _polygons_from_mask(mask, origin)
    _stamp_visual_metadata(polygons, extracted)
    for p in extracted:
        _refresh_metrics(p)
    extracted = drop_orphan_holes(extracted)
    return extracted

def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    n = len(points)
    return (sx / n, sy / n)
