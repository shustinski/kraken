"""Testable geometric post-processing for manual vector overlays (paths / editor).

Uses existing NumPy/OpenCV stack (same as graphics.geometry raster helpers).
See :class:`VectorGeometrySettings` for defaults and knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot, pi, radians

import cv2
import numpy as np

from ..domain import PolygonData, compute_polygon_metrics
from ..domain.polygon_ring import is_valid_closed_polygon_ring


def _polygon_contains_point(poly: PolygonData, point: tuple[float, float]) -> bool:
    from ..graphics.geometry import _polygon_contains_point as impl

    return impl(poly, point)


def _mask_helpers():
    from ..graphics.geometry import (
        _bbox_from_points,
        _polygons_from_mask,
        _render_polygon_collection_on_mask,
        _union_bbox,
    )

    return _bbox_from_points, _polygons_from_mask, _render_polygon_collection_on_mask, _union_bbox


@dataclass(slots=True)
class VectorGeometrySettings:
    """Editor / frame-sync vector cleanup (independent of mask extraction settings)."""

    clip_to_frame_on_sync: bool = True
    #: Minimum oriented area (px²) for an outer ring; excludes vias (`category=="via"` or `shape_hint=="box"`).
    min_outer_area_px2: float = 9.0
    #: Holes smaller than this area are filled; ``0`` disables (default: do not auto-fill holes during edit).
    min_hole_area_to_remove_px2: float = 0.0
    #: Merge conductors whose filled regions intersect after edits (vertex / polygon moves).
    merge_overlapping_on_edit: bool = True
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


def drop_polygons_invalid_points(polygons: list[PolygonData]) -> list[PolygonData]:
    keep: list[PolygonData] = []
    for p in polygons:
        if len(p.points) < 3:
            continue
        if any(not _point_finite(pt) for pt in p.points):
            continue
        keep.append(p)
    return keep


def filter_simple_valid_polygons(polygons: list[PolygonData]) -> list[PolygonData]:
    # Keep mask-extracted rings even when :func:`is_valid_closed_polygon_ring` rejects them,
    # otherwise hierarchy (outer + holes) silently collapses during editor post-process.
    return [p for p in polygons if len(p.points) >= 3]


def dissolve_small_holes(polygons: list[PolygonData], min_area_px2: float) -> list[PolygonData]:
    if min_area_px2 <= 0.0:
        return polygons
    return [p.clone() for p in polygons if not (p.is_hole and abs(float(p.area)) < float(min_area_px2))]


def drop_orphan_holes(polygons: list[PolygonData]) -> list[PolygonData]:
    ids = {p.id for p in polygons}
    return [p for p in polygons if p.parent_id is None or p.parent_id in ids]


def drop_small_outer_polygons(polygons: list[PolygonData], min_area_px2: float) -> list[PolygonData]:
    if min_area_px2 <= 0.0:
        return polygons
    def _keep_outer(poly: PolygonData) -> bool:
        if poly.is_hole:
            return True
        if poly.category == "via" or poly.shape_hint == "box":
            return True
        return abs(float(poly.area)) >= float(min_area_px2)

    drop_ids = {p.id for p in polygons if not _keep_outer(p)}
    survivors: list[PolygonData] = []
    for p in polygons:
        if p.id in drop_ids:
            continue
        if p.parent_id is not None and p.parent_id in drop_ids:
            continue
        survivors.append(p)
    return drop_orphan_holes(survivors)


def drop_triangle_outer_artifacts(
    polygons: list[PolygonData],
    enabled: bool,
    *,
    min_outer_area_px2: float = 0.0,
) -> list[PolygonData]:
    if not enabled:
        return polygons
    drop_ids: set[int] = set()
    threshold = float(min_outer_area_px2)
    for p in polygons:
        if p.is_hole or len(p.points) != 3 or p.category == "via" or p.shape_hint == "box":
            continue
        if p.shape_hint == "manual_outline":
            continue
        if threshold > 0.0:
            area_abs = abs(float(getattr(p, "area", 0.0) or 0.0))
            if area_abs <= 0.0:
                area_abs = abs(float(compute_polygon_metrics(p.points)[0]))
            if area_abs >= threshold:
                continue
        drop_ids.add(p.id)
    survivors = [p for p in polygons if p.id not in drop_ids and (p.parent_id is None or p.parent_id not in drop_ids)]
    return drop_orphan_holes(survivors)


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
    safety_cap = max(256, len(points) * len(points))
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
        q = p.clone()
        new_pts = remove_spikes_from_polygon_ring([(float(x), float(y)) for x, y in q.points], min_interior_angle_deg)
        if len(new_pts) < 3 or not is_valid_closed_polygon_ring(new_pts):
            out.append(p.clone())
            continue
        q.points = new_pts
        _refresh_metrics(q)
        out.append(q)
    out = drop_orphan_holes(out)
    return out


def clip_polygons_to_frame_raster(polygons: list[PolygonData], frame_width: int, frame_height: int) -> list[PolygonData]:
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


def _stamp_visual_metadata(reference: list[PolygonData], target: list[PolygonData]) -> None:
    for poly in target:
        c = _centroid(poly.points)
        best: PolygonData | None = None
        best_area = float("inf")
        for cand in reference:
            if cand.is_hole != poly.is_hole:
                continue
            if not cand.points:
                continue
            bx, by, bw, bh = cand.bbox
            if not (bx <= c[0] <= bx + bw and by <= c[1] <= by + bh):
                continue
            if _polygon_contains_point(cand, c):
                a = abs(float(cand.area))
                if best is None or a < best_area:
                    best = cand
                    best_area = a
        if best is None:
            continue
        poly.category = str(best.category)
        poly.shape_hint = str(best.shape_hint)
        poly.reject_reason = str(best.reject_reason)


class _UnionFind:
    def __init__(self, items: list[int]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: int) -> int:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while item != root:
            nxt = self._parent[item]
            self._parent[item] = root
            item = nxt
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _collect_family(polygons: list[PolygonData], root_id: int) -> list[PolygonData]:
    by_id = {p.id: p for p in polygons}
    out: list[PolygonData] = []
    pending = [root_id]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen or pid not in by_id:
            continue
        seen.add(pid)
        out.append(by_id[pid])
        for p in polygons:
            if p.parent_id == pid:
                pending.append(p.id)
    return out


def _family_bbox(polygons: list[PolygonData], root_id: int) -> tuple[int, int, int, int]:
    _bbox_from_points, *_rest = _mask_helpers()
    pts: list[tuple[float, float]] = []
    for p in _collect_family(polygons, root_id):
        pts.extend(p.points)
    if not pts:
        return (0, 0, 1, 1)
    return _bbox_from_points(pts, padding=2)


def _families_mask_overlap(polygons: list[PolygonData], root_a: int, root_b: int) -> bool:
    _, _, _render_polygon_collection_on_mask, _union_bbox = _mask_helpers()
    if root_a == root_b:
        return False
    bbox = _union_bbox([_family_bbox(polygons, root_a), _family_bbox(polygons, root_b)])
    x, y, w, h = bbox
    mask1 = np.zeros((max(1, h), max(1, w)), dtype=np.uint8)
    mask2 = np.zeros_like(mask1)
    fam1 = _collect_family(polygons, root_a)
    fam2 = _collect_family(polygons, root_b)
    _render_polygon_collection_on_mask(mask1, fam1, (x, y))
    _render_polygon_collection_on_mask(mask2, fam2, (x, y))
    return bool(np.any(cv2.bitwise_and(mask1, mask2)))


def merge_overlapping_root_families(polygons: list[PolygonData]) -> list[PolygonData]:
    _bbox_from_points, _polygons_from_mask, _render_polygon_collection_on_mask, _union_bbox = _mask_helpers()
    roots = [p.id for p in polygons if p.parent_id is None]
    if len(roots) < 2:
        return polygons
    uf = _UnionFind(roots)
    ordered = sorted(roots)
    for idx, ra in enumerate(ordered):
        for rb in ordered[idx + 1 :]:
            if _families_mask_overlap(polygons, ra, rb):
                uf.union(ra, rb)
    clusters: dict[int, set[int]] = {}
    for r in roots:
        root = uf.find(r)
        clusters.setdefault(root, set()).add(r)
    merged_roots = {leader for leader, members in clusters.items() if len(members) > 1}
    if not merged_roots:
        return polygons
    consumed_poly_ids: set[int] = set()
    survivors: list[PolygonData] = []
    rebuilt: list[list[PolygonData]] = []

    for leader in sorted(merged_roots):
        members = clusters[leader]
        combined: list[PolygonData] = []
        poly_ids_in_cluster: set[int] = set()
        for mr in sorted(members):
            fam = _collect_family(polygons, mr)
            combined.extend([p.clone() for p in fam])
            poly_ids_in_cluster.update(p.id for p in fam)
        rebuilt.append(poly_ids_in_cluster)
        consumed_poly_ids |= poly_ids_in_cluster

        bbox = _bbox_from_points(
            [(x, y) for p in combined for x, y in p.points],
            padding=4,
        )
        xo, yo, ww, hh = bbox
        mask = np.zeros((max(1, hh), max(1, ww)), dtype=np.uint8)
        _render_polygon_collection_on_mask(mask, combined, (xo, yo))
        merged = _polygons_from_mask(mask, (xo, yo))
        _stamp_visual_metadata(combined, merged)
        for p in merged:
            _refresh_metrics(p)
        survivors.extend(merged)

    leftover_ids = {p.id for p in polygons} - consumed_poly_ids
    leftovers = drop_orphan_holes([p.clone() for p in polygons if p.id in leftover_ids])

    merged_all = drop_orphan_holes(leftovers + survivors)
    return merged_all


def _polygons_topo_signature(polygons: list[PolygonData]) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for p in sorted(polygons, key=lambda q: q.id):
        rows.append(
            (
                p.id,
                bool(p.is_hole),
                p.parent_id,
                str(p.category),
                str(p.shape_hint),
                tuple((round(x, 4), round(y, 4)) for x, y in p.points),
            )
        )
    return tuple(rows)


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
    work = filter_simple_valid_polygons(work)
    for p in work:
        _refresh_metrics(p)

    fw, fh = frame_width_height or (0, 0)
    if settings.clip_to_frame_on_sync and fw > 1 and fh > 1:
        work = clip_polygons_to_frame_raster(work, fw, fh)

    work = dissolve_small_holes(work, settings.min_hole_area_to_remove_px2)
    work = apply_spike_removal_all(work, settings.min_spike_interior_angle_deg)
    work = filter_simple_valid_polygons(work)

    work = drop_small_outer_polygons(work, settings.min_outer_area_px2)
    work = drop_triangle_outer_artifacts(
        work,
        settings.drop_three_vertex_triangle_artifacts,
        min_outer_area_px2=settings.min_outer_area_px2,
    )

    work = drop_polygons_invalid_points(work)
    work = filter_simple_valid_polygons(work)

    if include_merge and settings.merge_overlapping_on_edit:
        work = merge_overlapping_root_families(work)

    work = drop_polygons_invalid_points(work)
    work = filter_simple_valid_polygons(work)

    work = drop_orphan_holes(work)
    changed = before != _polygons_topo_signature(work)
    return work, changed


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
    work = filter_simple_valid_polygons(work)
    if settings.clip_to_frame_on_sync and frame_width > 1 and frame_height > 1:
        work = clip_polygons_to_frame_raster(work, frame_width, frame_height)
    work = dissolve_small_holes(work, settings.min_hole_area_to_remove_px2)
    work = apply_spike_removal_all(work, settings.min_spike_interior_angle_deg)
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


def apply_vertex_position_to_clone(
    polygons: list[PolygonData],
    polygon_id: int,
    vertex_index: int,
    new_point: tuple[float, float],
) -> list[PolygonData]:
    work = [p.clone() for p in polygons]
    by_id = {p.id: p for p in work}
    target = by_id.get(polygon_id)
    if target is None or vertex_index < 0 or vertex_index >= len(target.points):
        return work
    pts = [(float(x), float(y)) for x, y in target.points]
    pts[vertex_index] = (float(new_point[0]), float(new_point[1]))
    target.points = pts
    if not is_valid_closed_polygon_ring(target.points):
        return [p.clone() for p in polygons]
    _refresh_metrics(target)
    return work


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
    target.points = [(float(x), float(y)) for x, y in new_points]
    if not is_valid_closed_polygon_ring(target.points):
        return [p.clone() for p in polygons]
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
