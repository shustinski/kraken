"""Testable geometric post-processing for manual vector overlays (paths / editor).

Uses existing NumPy/OpenCV stack (same as graphics.geometry raster helpers).
See :class:`VectorGeometrySettings` for defaults and knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot, pi, radians

import cv2
import numpy as np

from ..domain import PolygonData, compute_polygon_metrics, integer_point, integer_points
from ..domain.polygon_ring import (
    TOPOLOGY_CHECK_MAX_VERTICES,
    collapse_redundant_polyline_vertices,
    is_valid_closed_polygon_ring,
    is_valid_closed_polygon_edge_move,
    is_valid_closed_polygon_vertex_move,
)


def _ring_ok_after_spike_removal(points: list[tuple[float, float]]) -> bool:
    if len(points) < 3:
        return False
    if len(points) > TOPOLOGY_CHECK_MAX_VERTICES:
        return True
    return is_valid_closed_polygon_ring(points)


def _polygon_contains_point(poly: PolygonData, point: tuple[float, float]) -> bool:
    from ..graphics.geometry import _polygon_contains_point as impl

    return impl(poly, point)


_MASK_HELPERS: tuple | None = None


def _mask_helpers():
    global _MASK_HELPERS
    if _MASK_HELPERS is None:
        from ..graphics.geometry import (
            _bbox_from_points,
            _polygons_from_mask,
            _render_polygon_collection_on_mask,
            _union_bbox,
        )

        _MASK_HELPERS = (
            _bbox_from_points,
            _polygons_from_mask,
            _render_polygon_collection_on_mask,
            _union_bbox,
        )
    return _MASK_HELPERS


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
    from ..graphics.brush_vector import ring_is_valid_for_polygon

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
    return [p.clone() for p in polygons if not (p.is_hole and _is_small_inner_area(p, min_area_px2))]


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


def _promoted_root_ids_before_reparent(polygons: list[PolygonData], removed_ids: set[int]) -> set[int]:
    return {
        polygon.id
        for polygon in polygons
        if polygon.parent_id in removed_ids and not _is_via_like(polygon)
    }


def _expand_roots_by_touching_bboxes(
    seed_roots: set[int],
    root_bboxes: dict[int, tuple[int, int, int, int]],
) -> set[int]:
    expanded = set(seed_roots)
    for seed in seed_roots:
        seed_bbox = root_bboxes.get(seed)
        if seed_bbox is None:
            continue
        for other, other_bbox in root_bboxes.items():
            if other in expanded:
                continue
            if _bboxes_intersect(seed_bbox, other_bbox):
                expanded.add(other)
    return expanded


def _merge_candidate_roots_after_removal(
    polygons: list[PolygonData],
    removed_ids: set[int],
    *,
    removed_polygons: list[PolygonData] | None = None,
) -> set[int] | None:
    """Roots that may overlap after deletions; ``None`` means check every root pair."""

    if not removed_ids:
        return None
    root_bboxes = _root_family_bboxes(polygons)
    seeds = set(_promoted_root_ids_before_reparent(polygons, removed_ids))
    if removed_polygons:
        touch_bboxes = [polygon.bbox for polygon in removed_polygons if polygon.bbox is not None]
        if touch_bboxes:
            _, _, _, _union_bbox = _mask_helpers()
            touch_union = _union_bbox(touch_bboxes)
            seeds.update(
                root_id
                for root_id, bbox in root_bboxes.items()
                if _bboxes_intersect(bbox, touch_union)
            )
    if not seeds:
        return set()
    return _expand_roots_by_touching_bboxes(seeds, root_bboxes)


def union_after_removing_polygon_ids(
    remaining: list[PolygonData],
    removed_ids: set[int],
    *,
    removed_polygons: list[PolygonData] | None = None,
) -> list[PolygonData]:
    """Reparent orphans, then dissolve conductor–conductor overlaps."""

    merge_candidates = _merge_candidate_roots_after_removal(
        remaining,
        removed_ids,
        removed_polygons=removed_polygons,
    )
    work = reparent_polygons_orphaned_by_ids(remaining, removed_ids)
    work = drop_orphan_holes(work)
    work = merge_overlapping_root_families(work, candidate_roots=merge_candidates)
    return collapse_redundant_vertices_in_polygons(work)


def apply_orphan_cleanup_after_removal(
    remaining: list[PolygonData],
    removed_ids: set[int],
) -> list[PolygonData]:
    """Reparent islands and drop orphan holes without overlap merge."""

    work = reparent_polygons_orphaned_by_ids(remaining, removed_ids)
    return drop_orphan_holes(work)


def apply_topology_repair_after_edit(
    polygons: list[PolygonData],
    *,
    changed_polygon_ids: set[int],
) -> list[PolygonData]:
    """Rebuild self-intersecting rings touched by a local edit (always run synchronously)."""

    work = dissolve_self_intersecting_polygons(
        polygons,
        changed_polygon_ids=changed_polygon_ids,
    )
    if not changed_polygon_ids:
        return collapse_redundant_vertices_in_polygons(work)
    original_ids = {polygon.id for polygon in polygons}
    collapse_ids = set(changed_polygon_ids)
    collapse_ids.update(polygon.id for polygon in work if polygon.id not in original_ids)
    work, _ = collapse_redundant_vertices_for_polygon_ids(work, collapse_ids)
    return work


def apply_overlap_merge_after_edit(
    polygons: list[PolygonData],
    *,
    removed_ids: set[int],
    removed_polygons: list[PolygonData] | None = None,
    changed_polygon_ids: set[int] | None = None,
) -> list[PolygonData]:
    """Merge conductor root families after holes/removals (may be deferred on bulk edits)."""

    if removed_ids:
        merge_candidates = _merge_candidate_roots_after_removal(
            polygons,
            removed_ids,
            removed_polygons=removed_polygons or [],
        )
        work = merge_overlapping_root_families(polygons, candidate_roots=merge_candidates)
        return collapse_redundant_vertices_in_polygons(work)
    if changed_polygon_ids:
        work = merge_overlapping_root_families_near_polygons(
            polygons,
            changed_polygon_ids,
            expand_touching_roots=False,
        )
        return collapse_redundant_vertices_in_polygons(work)
    return collapse_redundant_vertices_in_polygons(polygons)


def _repair_still_invalid_ring_families(
    polygons: list[PolygonData],
    *,
    seed_polygon_ids: set[int],
) -> list[PolygonData]:
    """Rebuild any families that still contain invalid rings after the main repair pass."""

    if not seed_polygon_ids:
        return polygons
    by_id = {polygon.id: polygon for polygon in polygons}
    invalid_ids: set[int] = set()
    for polygon_id in seed_polygon_ids:
        root_id = _root_id_for_polygon(by_id, polygon_id)
        if root_id is None:
            continue
        for member in _collect_family(polygons, root_id):
            if member.description_is_invalid():
                invalid_ids.add(member.id)
    if not invalid_ids:
        return polygons
    return dissolve_self_intersecting_polygons(polygons, changed_polygon_ids=invalid_ids)


def apply_delete_area_geometry_repair(
    polygons: list[PolygonData],
    settings: VectorGeometrySettings,
    *,
    changed_polygon_ids: set[int],
    removed_ids: set[int],
    removed_polygons: list[PolygonData] | None = None,
) -> list[PolygonData]:
    """Full synchronous repair after delete-area: topology, orphans, merge, final invalid pass."""

    seed_ids = set(changed_polygon_ids) | set(removed_ids)
    work = apply_topology_repair_after_edit(
        polygons,
        changed_polygon_ids=changed_polygon_ids,
    )
    if removed_ids:
        work = apply_orphan_cleanup_after_removal(work, removed_ids)
    if settings.merge_overlapping_on_edit and seed_ids:
        work = apply_overlap_merge_after_edit(
            work,
            removed_ids=removed_ids,
            removed_polygons=removed_polygons,
            changed_polygon_ids=changed_polygon_ids if not removed_ids else None,
        )
    return _repair_still_invalid_ring_families(work, seed_polygon_ids=seed_ids)


def apply_heavy_geometry_repair_after_edit(
    polygons: list[PolygonData],
    *,
    changed_polygon_ids: set[int],
    removed_ids: set[int],
    removed_polygons: list[PolygonData] | None = None,
) -> list[PolygonData]:
    """Dissolve invalid rings and merge conductors after a local edit."""

    work = apply_topology_repair_after_edit(
        polygons,
        changed_polygon_ids=changed_polygon_ids,
    )
    return apply_overlap_merge_after_edit(
        work,
        removed_ids=removed_ids,
        removed_polygons=removed_polygons,
    )


def should_defer_overlap_merge_after_edit(
    settings: VectorGeometrySettings,
    *,
    changed_polygon_ids: set[int],
    removed_ids: set[int],
) -> bool:
    """Whether overlap merge may run after commit instead of blocking the edit."""

    if not removed_ids:
        return False
    affected = len(changed_polygon_ids | removed_ids)
    threshold = max(1, int(settings.deferred_geometry_repair_affected_threshold))
    return affected >= threshold


def should_defer_geometry_repair(
    settings: VectorGeometrySettings,
    *,
    changed_polygon_ids: set[int],
    removed_ids: set[int],
) -> bool:
    """Backward-compatible alias for overlap-merge deferral."""

    return should_defer_overlap_merge_after_edit(
        settings,
        changed_polygon_ids=changed_polygon_ids,
        removed_ids=removed_ids,
    )


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


def _relabel_polygon_ids(polygons: list[PolygonData], *, start_id: int) -> list[PolygonData]:
    """Assign unique ids starting at ``start_id``, remapping ``parent_id`` within the list.

    Mask / shapely rebuild helpers always emit ids from 1. Without this remapping,
    repaired rings collide with untouched polygons and wipe them in the editor.
    """

    if not polygons:
        return []
    old_to_new: dict[int, int] = {}
    next_id = max(1, int(start_id))
    for polygon in polygons:
        if polygon.id in old_to_new:
            continue
        old_to_new[polygon.id] = next_id
        next_id += 1
    relabeled: list[PolygonData] = []
    for polygon in polygons:
        clone = polygon.clone()
        clone.id = old_to_new[polygon.id]
        if clone.parent_id is not None:
            clone.parent_id = old_to_new.get(clone.parent_id)
        relabeled.append(clone)
    return relabeled


def _next_polygon_id_after(polygons: list[PolygonData]) -> int:
    if not polygons:
        return 1
    return max(polygon.id for polygon in polygons) + 1


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


def _children_by_parent(polygons: list[PolygonData]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for polygon in polygons:
        parent_id = polygon.parent_id
        if parent_id is not None:
            children.setdefault(parent_id, []).append(polygon.id)
    return children


def _pad_bbox(bbox: tuple[int, int, int, int], padding: int) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    return (x - padding, y - padding, width + 2 * padding, height + 2 * padding)


def _root_id_map(polygons: list[PolygonData]) -> dict[int, int]:
    by_id = {polygon.id: polygon for polygon in polygons}
    roots: dict[int, int] = {}

    def resolve(polygon_id: int) -> int:
        cached = roots.get(polygon_id)
        if cached is not None:
            return cached
        seen: set[int] = set()
        chain: list[int] = []
        current = polygon_id
        while current in by_id:
            cached = roots.get(current)
            if cached is not None:
                for node in chain:
                    roots[node] = cached
                return cached
            if current in seen:
                for node in chain:
                    roots[node] = current
                return current
            seen.add(current)
            chain.append(current)
            parent_id = by_id[current].parent_id
            if parent_id is None:
                for node in chain:
                    roots[node] = current
                return current
            current = parent_id
        for node in chain:
            roots[node] = polygon_id
        return polygon_id

    for polygon in polygons:
        resolve(polygon.id)
    return roots


def _root_family_bboxes(polygons: list[PolygonData]) -> dict[int, tuple[int, int, int, int]]:
    """Map conductor root id → padded union bbox of all family members."""

    _, _, _, _union_bbox = _mask_helpers()
    root_map = _root_id_map(polygons)
    conductor_roots = {
        polygon.id
        for polygon in polygons
        if polygon.parent_id is None and not polygon.is_hole and not _is_via_like(polygon)
    }
    grouped: dict[int, list[tuple[int, int, int, int]]] = {
        root_id: [] for root_id in conductor_roots
    }
    for polygon in polygons:
        root_id = root_map.get(polygon.id)
        if root_id is None or root_id not in grouped or polygon.bbox is None:
            continue
        grouped[root_id].append(polygon.bbox)
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for root_id, boxes in grouped.items():
        if not boxes:
            bboxes[root_id] = (0, 0, 1, 1)
        else:
            bboxes[root_id] = _pad_bbox(_union_bbox(boxes), 2)
    return bboxes


def _collect_family(
    polygons: list[PolygonData],
    root_id: int,
    *,
    by_id: dict[int, PolygonData] | None = None,
    children: dict[int, list[int]] | None = None,
) -> list[PolygonData]:
    by_id = by_id or {polygon.id: polygon for polygon in polygons}
    children = children or _children_by_parent(polygons)
    out: list[PolygonData] = []
    pending = [root_id]
    seen: set[int] = set()
    while pending:
        polygon_id = pending.pop()
        if polygon_id in seen or polygon_id not in by_id:
            continue
        seen.add(polygon_id)
        out.append(by_id[polygon_id])
        pending.extend(children.get(polygon_id, ()))
    return out


def _family_bbox(
    polygons: list[PolygonData],
    root_id: int,
    *,
    by_id: dict[int, PolygonData] | None = None,
    children: dict[int, list[int]] | None = None,
    root_bboxes: dict[int, tuple[int, int, int, int]] | None = None,
) -> tuple[int, int, int, int]:
    if root_bboxes is not None:
        cached = root_bboxes.get(root_id)
        if cached is not None:
            return cached
    _, _, _, _union_bbox = _mask_helpers()
    family = _collect_family(polygons, root_id, by_id=by_id, children=children)
    member_bboxes = [polygon.bbox for polygon in family if polygon.bbox is not None]
    if not member_bboxes:
        return (0, 0, 1, 1)
    return _pad_bbox(_union_bbox(member_bboxes), 2)


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
    return int(np.count_nonzero(cv2.bitwise_and(mask1, mask2))) > 1


def _families_mask_overlap_cached(
    root_a: int,
    root_b: int,
    *,
    families: dict[int, list[PolygonData]],
    bboxes: dict[int, tuple[int, int, int, int]],
) -> bool:
    _, _, _render_polygon_collection_on_mask, _union_bbox = _mask_helpers()
    if root_a == root_b:
        return False
    bbox_a = bboxes[root_a]
    bbox_b = bboxes[root_b]
    if not _bboxes_intersect(bbox_a, bbox_b):
        return False
    x, y, w, h = _union_bbox([bbox_a, bbox_b])
    mask1 = np.zeros((max(1, h), max(1, w)), dtype=np.uint8)
    mask2 = np.zeros_like(mask1)
    _render_polygon_collection_on_mask(mask1, families[root_a], (x, y))
    _render_polygon_collection_on_mask(mask2, families[root_b], (x, y))
    # Require >1 pixel so shared vertices / edge touches from rasterization do not count.
    return int(np.count_nonzero(cv2.bitwise_and(mask1, mask2))) > 1


def _merge_candidate_roots_near_polygons(
    polygons: list[PolygonData],
    polygon_ids: set[int],
    *,
    expand_touching_roots: bool = True,
) -> set[int]:
    if not polygon_ids:
        return set()
    by_id = {polygon.id: polygon for polygon in polygons}
    root_bboxes = _root_family_bboxes(polygons)
    seeds: set[int] = set()
    touch_bboxes: list[tuple[int, int, int, int]] = []
    for polygon_id in polygon_ids:
        root_id = _root_id_for_polygon(by_id, polygon_id)
        if root_id is not None:
            seeds.add(root_id)
        polygon = by_id.get(polygon_id)
        if polygon is not None and polygon.bbox is not None:
            touch_bboxes.append(polygon.bbox)
    if touch_bboxes:
        _, _, _, _union_bbox = _mask_helpers()
        touch_union = _union_bbox(touch_bboxes)
        seeds.update(
            root_id
            for root_id, bbox in root_bboxes.items()
            if _bboxes_intersect(bbox, touch_union)
        )
    if not seeds:
        return set()
    if expand_touching_roots:
        return _expand_roots_by_touching_bboxes(seeds, root_bboxes)
    return seeds


def merge_overlapping_root_families_near_polygons(
    polygons: list[PolygonData],
    polygon_ids: set[int] | int,
    *,
    expand_touching_roots: bool = True,
) -> list[PolygonData]:
    """Merge only root families near edited polygons instead of scanning the full layer."""

    ids = {polygon_ids} if isinstance(polygon_ids, int) else set(polygon_ids)
    candidate_roots = _merge_candidate_roots_near_polygons(
        polygons,
        ids,
        expand_touching_roots=expand_touching_roots,
    )
    return merge_overlapping_root_families(polygons, candidate_roots=candidate_roots)


def merge_overlapping_root_families(
    polygons: list[PolygonData],
    *,
    candidate_roots: set[int] | None = None,
) -> list[PolygonData]:
    _bbox_from_points, _polygons_from_mask, _render_polygon_collection_on_mask, _union_bbox = _mask_helpers()
    roots = [
        polygon.id
        for polygon in polygons
        if polygon.parent_id is None and not polygon.is_hole and not _is_via_like(polygon)
    ]
    if len(roots) < 2:
        return polygons
    if candidate_roots is not None and not candidate_roots:
        return polygons
    uf = _UnionFind(roots)
    ordered = sorted(roots)
    by_id = {polygon.id: polygon for polygon in polygons}
    children = _children_by_parent(polygons)
    bboxes = _root_family_bboxes(polygons)
    family_roots = set(roots if candidate_roots is None else candidate_roots)
    if candidate_roots is not None:
        for root_id in roots:
            if root_id in family_roots:
                continue
            root_bbox = bboxes[root_id]
            if any(_bboxes_intersect(root_bbox, bboxes[candidate_root]) for candidate_root in candidate_roots):
                family_roots.add(root_id)
    families = {
        root_id: _collect_family(polygons, root_id, by_id=by_id, children=children)
        for root_id in family_roots
    }
    for idx, root_a in enumerate(ordered):
        for root_b in ordered[idx + 1 :]:
            if candidate_roots is not None and root_a not in candidate_roots and root_b not in candidate_roots:
                continue
            if not _bboxes_intersect(bboxes[root_a], bboxes[root_b]):
                continue
            for root_id in (root_a, root_b):
                if root_id not in families:
                    families[root_id] = _collect_family(polygons, root_id, by_id=by_id, children=children)
            if _families_mask_overlap_cached(root_a, root_b, families=families, bboxes=bboxes):
                uf.union(root_a, root_b)
    clusters: dict[int, set[int]] = {}
    for root_id in roots:
        leader = uf.find(root_id)
        clusters.setdefault(leader, set()).add(root_id)
    merged_roots = {leader for leader, members in clusters.items() if len(members) > 1}
    if not merged_roots:
        return polygons
    consumed_poly_ids: set[int] = set()
    survivors: list[PolygonData] = []
    next_id = _next_polygon_id_after(polygons)

    for leader in sorted(merged_roots):
        members = clusters[leader]
        combined: list[PolygonData] = []
        poly_ids_in_cluster: set[int] = set()
        for mr in sorted(members):
            fam = _collect_family(polygons, mr)
            combined.extend([p.clone() for p in fam])
            poly_ids_in_cluster.update(p.id for p in fam)
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
        merged = _relabel_polygon_ids(merged, start_id=next_id)
        if merged:
            next_id = _next_polygon_id_after(merged)
        survivors.extend(merged)

    leftover_ids = {p.id for p in polygons} - consumed_poly_ids
    leftovers = drop_orphan_holes([p.clone() for p in polygons if p.id in leftover_ids])
    return drop_orphan_holes(leftovers + survivors)


def _rebuild_family_from_filled_mask(family: list[PolygonData]) -> list[PolygonData]:
    _bbox_from_points, _polygons_from_mask, _render_polygon_collection_on_mask, _union_bbox = _mask_helpers()
    del _union_bbox
    points = [(x_coord, y_coord) for polygon in family for x_coord, y_coord in polygon.points]
    if not points:
        return []
    origin_x, origin_y, width, height = _bbox_from_points(points, padding=4)
    mask = np.zeros((max(1, height), max(1, width)), dtype=np.uint8)
    _render_polygon_collection_on_mask(mask, [polygon.clone() for polygon in family], (origin_x, origin_y))
    extracted = _polygons_from_mask(mask, (origin_x, origin_y))
    _stamp_visual_metadata(family, extracted)
    for polygon in extracted:
        _refresh_metrics(polygon)
    return extracted


def _polygon_ring_needs_dissolve(polygon: PolygonData) -> bool:
    return polygon.description_is_invalid()


def _polygon_ring_has_keyhole_bridge(polygon: PolygonData) -> bool:
    return (not polygon.is_hole) and polygon.description_invalid_reason() == "repeated_vertex"


def _polygon_ring_needs_explicit_topology_repair(polygon: PolygonData) -> bool:
    """True for bowties or CIF keyholes when the user explicitly requests repair."""

    return _polygon_ring_needs_dissolve(polygon) or _polygon_ring_has_keyhole_bridge(polygon)


def dissolve_self_intersecting_polygons(
    polygons: list[PolygonData],
    *,
    include_keyhole_bridges: bool = False,
    changed_polygon_ids: set[int] | None = None,
) -> list[PolygonData]:
    """Rebuild rings whose new edges cross existing ones into a simple filled union.

    By default only true invalids (e.g. bowties) are rewritten. Pass
    ``include_keyhole_bridges=True`` for explicit repair that also splits CIF
    keyhole ``repeated_vertex`` bridges into outer+hole children.

    When ``changed_polygon_ids`` is set, only scan and rebuild root families
    touched by those edits (unchanged rings keep their cached validity).
    """

    def _needs(polygon: PolygonData) -> bool:
        if include_keyhole_bridges:
            return _polygon_ring_needs_explicit_topology_repair(polygon)
        return _polygon_ring_needs_dissolve(polygon)

    roots_to_process: set[int] | None = None
    if changed_polygon_ids is not None:
        if not changed_polygon_ids:
            return polygons
        polygons_by_id = {polygon.id: polygon for polygon in polygons}
        roots_to_process = {
            root_id
            for polygon_id in changed_polygon_ids
            if (root_id := _root_id_for_polygon(polygons_by_id, polygon_id)) is not None
        }
        if not roots_to_process:
            return polygons
        if not any(
            _needs(polygons_by_id[polygon_id])
            for polygon_id in changed_polygon_ids
            if polygon_id in polygons_by_id
        ):
            return polygons
    elif not any(_needs(polygon) for polygon in polygons):
        return polygons

    roots = [polygon.id for polygon in polygons if polygon.parent_id is None]
    consumed_ids: set[int] = set()
    rebuilt: list[PolygonData] = []
    next_id = _next_polygon_id_after(polygons)
    for root_id in roots:
        if roots_to_process is not None and root_id not in roots_to_process:
            continue
        family = _collect_family(polygons, root_id)
        if changed_polygon_ids is not None:
            family_changed = [
                polygon for polygon in family if polygon.id in changed_polygon_ids
            ]
            if not family_changed or not any(_needs(polygon) for polygon in family_changed):
                continue
        elif not any(_needs(polygon) for polygon in family):
            continue
        extracted = _dissolve_family_to_simple_rings(family)
        if not extracted:
            continue
        _stamp_visual_metadata(family, extracted)
        for polygon in extracted:
            _refresh_metrics(polygon)
        extracted = _relabel_polygon_ids(extracted, start_id=next_id)
        next_id = _next_polygon_id_after(extracted)
        rebuilt.extend(extracted)
        consumed_ids.update(polygon.id for polygon in family)

    if not consumed_ids:
        return polygons
    leftovers = [polygon.clone() for polygon in polygons if polygon.id not in consumed_ids]
    return drop_orphan_holes(leftovers + rebuilt)


def _repair_family_by_keyhole_split(family: list[PolygonData]) -> list[PolygonData] | None:
    """Split CIF keyhole polylines into outer+holes without reshaping via make_valid.

    Shapely dissolve changes the filled outline; keyhole extraction keeps the authored
    rings and only cuts the bridge that made the description invalid.
    """

    from ..serializers import _split_linked_polygon_rings

    if not any(
        (not polygon.is_hole) and polygon.description_invalid_reason() == "repeated_vertex"
        for polygon in family
    ):
        return None

    current = [polygon.clone() for polygon in family]
    changed = False
    for _pass in range(8):
        next_id = max((polygon.id for polygon in current), default=0) + 1
        rebuilt: list[PolygonData] = []
        progress = False
        for polygon in current:
            if polygon.is_hole or polygon.description_invalid_reason() != "repeated_vertex":
                rebuilt.append(polygon)
                continue
            outer_points, hole_rings = _split_linked_polygon_rings(polygon.points)
            if not hole_rings:
                rebuilt.append(polygon)
                continue
            progress = True
            changed = True
            area, perimeter, bbox = compute_polygon_metrics(outer_points)
            outer = polygon.clone()
            outer.points = list(outer_points)
            outer.area = float(area)
            outer.perimeter = float(perimeter)
            outer.bbox = bbox
            rebuilt.append(outer)
            for hole_points in hole_rings:
                area, perimeter, bbox = compute_polygon_metrics(hole_points)
                rebuilt.append(
                    PolygonData(
                        id=next_id,
                        points=list(hole_points),
                        is_hole=True,
                        parent_id=outer.id,
                        category=str(polygon.category),
                        shape_hint=str(polygon.shape_hint),
                        area=float(area),
                        perimeter=float(perimeter),
                        bbox=bbox,
                    )
                )
                next_id += 1
        current = rebuilt
        if not progress:
            break

    if not changed:
        return None
    if any(_polygon_ring_needs_explicit_topology_repair(polygon) for polygon in current):
        return None
    return current


def _dissolve_family_to_simple_rings(family: list[PolygonData]) -> list[PolygonData]:
    from ..graphics.brush_vector import region_geometry, shapely_to_polygon_data_list

    keyhole_repaired = _repair_family_by_keyhole_split(family)
    if keyhole_repaired is not None:
        return keyhole_repaired

    by_id = {polygon.id: polygon for polygon in family}
    extracted = shapely_to_polygon_data_list(region_geometry(by_id, [polygon.id for polygon in family]))
    if not extracted:
        extracted = shapely_to_polygon_data_list(_make_valid_union_of_family_rings(family))
    if not extracted:
        extracted = _rebuild_family_from_filled_mask(family)
    if extracted and any(_polygon_ring_needs_explicit_topology_repair(polygon) for polygon in extracted):
        repaired = shapely_to_polygon_data_list(_make_valid_union_of_family_rings(extracted))
        if repaired:
            extracted = repaired
    return extracted


def _make_valid_union_of_family_rings(family: list[PolygonData]):
    from shapely import make_valid, unary_union
    from shapely.geometry import Polygon as ShapelyPolygon

    parts = []
    for polygon in family:
        if polygon.is_hole:
            continue
        shell = [(float(x_coord), float(y_coord)) for x_coord, y_coord in polygon.points]
        interiors = [
            [(float(x_coord), float(y_coord)) for x_coord, y_coord in hole.points]
            for hole in family
            if hole.is_hole and hole.parent_id == polygon.id
        ]
        try:
            geom = ShapelyPolygon(shell, interiors) if interiors else ShapelyPolygon(shell)
        except (ValueError, TypeError):
            geom = ShapelyPolygon(shell)
        repaired = unary_union(make_valid(geom))
        if not repaired.is_empty:
            parts.append(repaired)
    if not parts:
        return ShapelyPolygon()
    return unary_union(parts)


def polygon_description_is_invalid(polygon: PolygonData) -> bool:
    return polygon.description_is_invalid()


def polygon_description_invalid_reason(polygon: PolygonData) -> str | None:
    return polygon.description_invalid_reason()


def overlapping_root_family_ids(polygons: list[PolygonData]) -> set[int]:
    """Return non-via root ids whose filled families mask-overlap at least one sibling root."""

    return overlapping_root_family_ids_near_roots(
        polygons,
        {
            polygon.id
            for polygon in polygons
            if polygon.parent_id is None and not polygon.is_hole and not _is_via_like(polygon)
        },
    )


def overlapping_root_family_ids_near_roots(
    polygons: list[PolygonData],
    candidate_roots: set[int],
) -> set[int]:
    """Return overlapping root ids among pairs where at least one root is in ``candidate_roots``."""

    if not candidate_roots:
        return set()
    roots = [
        polygon.id
        for polygon in polygons
        if polygon.parent_id is None and not polygon.is_hole and not _is_via_like(polygon)
    ]
    if len(roots) < 2:
        return set()
    ordered = sorted(roots)
    by_id = {polygon.id: polygon for polygon in polygons}
    children = _children_by_parent(polygons)
    families = {
        root_id: _collect_family(polygons, root_id, by_id=by_id, children=children)
        for root_id in ordered
    }
    bboxes = _root_family_bboxes(polygons)
    overlapping: set[int] = set()
    for idx, root_a in enumerate(ordered):
        for root_b in ordered[idx + 1 :]:
            if root_a not in candidate_roots and root_b not in candidate_roots:
                continue
            if _families_mask_overlap_cached(root_a, root_b, families=families, bboxes=bboxes):
                overlapping.add(root_a)
                overlapping.add(root_b)
    return overlapping


def overlap_check_roots_for_layer_patch(
    polygons: list[PolygonData],
    removed_polygons: list[PolygonData],
    added_or_changed: list[PolygonData],
) -> set[int]:
    """Return root ids whose geometry may need overlap repair after a local edit."""

    touch_bboxes = [
        polygon.bbox
        for polygon in (*removed_polygons, *added_or_changed)
        if polygon.bbox is not None
    ]
    if not touch_bboxes:
        return set()
    root_bboxes = _root_family_bboxes(polygons)
    if not root_bboxes:
        return set()
    _, _, _, _union_bbox = _mask_helpers()
    touch_union = _union_bbox(touch_bboxes)
    return {
        root_id
        for root_id, bbox in root_bboxes.items()
        if _bboxes_intersect(bbox, touch_union)
    }


def apply_overlap_repair_patch(
    existing: dict[int, list[str]],
    polygons: list[PolygonData],
    overlap_check_roots: set[int],
) -> dict[int, list[str]]:
    """Recompute overlapping repair marks for roots near a local edit."""

    reasons_by_id = {polygon_id: list(reason_codes) for polygon_id, reason_codes in existing.items()}

    def _add(polygon_id: int, reason: str) -> None:
        existing_reasons = reasons_by_id.setdefault(polygon_id, [])
        if reason not in existing_reasons:
            existing_reasons.append(reason)

    for root_id in overlap_check_roots:
        for member in _collect_family(polygons, root_id):
            reason_codes = reasons_by_id.get(member.id)
            if reason_codes is None or "overlapping" not in reason_codes:
                continue
            filtered = [reason for reason in reason_codes if reason != "overlapping"]
            if filtered:
                reasons_by_id[member.id] = filtered
            else:
                reasons_by_id.pop(member.id, None)

    overlapping_roots = overlapping_root_family_ids_near_roots(polygons, overlap_check_roots)
    for root_id in overlapping_roots:
        for member in _collect_family(polygons, root_id):
            _add(member.id, "overlapping")

    return reasons_by_id


def patch_polygons_needing_repair(
    existing: dict[int, list[str]],
    polygons: list[PolygonData],
    *,
    removed_ids: set[int],
    removed_polygons: list[PolygonData],
    added_or_changed: list[PolygonData],
    settings: VectorGeometrySettings,
    include_overlap: bool = True,
) -> dict[int, list[str]]:
    """Update a repair map after an incremental polygon-layer edit."""

    reasons_by_id = {
        polygon_id: list(reason_codes)
        for polygon_id, reason_codes in existing.items()
        if polygon_id not in removed_ids
    }

    def _add(polygon_id: int, reason: str) -> None:
        existing_reasons = reasons_by_id.setdefault(polygon_id, [])
        if reason not in existing_reasons:
            existing_reasons.append(reason)

    min_outer = float(settings.min_outer_area_px2)
    min_hole = float(settings.min_hole_area_to_remove_px2)
    for polygon in added_or_changed:
        if polygon.description_is_invalid():
            ring_reason = polygon.description_invalid_reason()
            if ring_reason is not None:
                _add(polygon.id, ring_reason)
        if polygon.is_hole:
            if min_hole > 0.0 and _is_small_inner_area(polygon, min_hole):
                _add(polygon.id, "small_hole")
            continue
        if _is_via_like(polygon):
            continue
        if min_outer > 0.0 and abs(float(polygon.area)) < min_outer:
            _add(polygon.id, "small_object")

    if not include_overlap:
        return reasons_by_id

    overlap_check_roots = overlap_check_roots_for_layer_patch(
        polygons,
        removed_polygons,
        added_or_changed,
    )
    return apply_overlap_repair_patch(reasons_by_id, polygons, overlap_check_roots)


def polygons_needing_repair(
    polygons: list[PolygonData],
    settings: VectorGeometrySettings,
    *,
    include_ring_reasons: bool = True,
    include_overlap: bool = True,
) -> dict[int, list[str]]:
    """Map polygon id → ordered unique repair reason codes (topology + area + overlap).

    Reason codes:
    - ``self_intersecting`` — ring-local invalid description (bowtie / crossing)
    - ``overlapping`` — filled root family intersects another conductor root
    - ``small_object`` — non-via outer below ``settings.min_outer_area_px2``
    - ``small_hole`` — hole below ``settings.min_hole_area_to_remove_px2``

    Note: CIF keyhole ``repeated_vertex`` is intentionally omitted here so auto red-mark
    and repair offers do not treat valid keyhole ``P`` rings as broken.
    """

    reasons_by_id: dict[int, list[str]] = {}

    def _add(polygon_id: int, reason: str) -> None:
        existing = reasons_by_id.setdefault(polygon_id, [])
        if reason not in existing:
            existing.append(reason)

    if include_ring_reasons:
        for polygon in polygons:
            if polygon.description_is_invalid():
                ring_reason = polygon.description_invalid_reason()
                if ring_reason is not None:
                    _add(polygon.id, ring_reason)

    if not include_overlap:
        min_outer = float(settings.min_outer_area_px2)
        min_hole = float(settings.min_hole_area_to_remove_px2)
        for polygon in polygons:
            if polygon.is_hole:
                if min_hole > 0.0 and _is_small_inner_area(polygon, min_hole):
                    _add(polygon.id, "small_hole")
                continue
            if _is_via_like(polygon):
                continue
            if min_outer > 0.0 and abs(float(polygon.area)) < min_outer:
                _add(polygon.id, "small_object")
        return reasons_by_id

    overlapping_roots = overlapping_root_family_ids(polygons)
    for root_id in overlapping_roots:
        for member in _collect_family(polygons, root_id):
            _add(member.id, "overlapping")

    min_outer = float(settings.min_outer_area_px2)
    min_hole = float(settings.min_hole_area_to_remove_px2)
    for polygon in polygons:
        if polygon.is_hole:
            if min_hole > 0.0 and _is_small_inner_area(polygon, min_hole):
                _add(polygon.id, "small_hole")
            continue
        if _is_via_like(polygon):
            continue
        if min_outer > 0.0 and abs(float(polygon.area)) < min_outer:
            _add(polygon.id, "small_object")

    return reasons_by_id


def summarize_invalid_polygon_description_reasons(
    polygons: list[PolygonData],
    settings: VectorGeometrySettings | None = None,
) -> list[tuple[str, int]]:
    """Count repair reason codes across polygons, preserving first-seen order.

    When ``settings`` is omitted, only ring-local invalid-description reasons are counted
    (backwards-compatible with topology-only callers). With settings, also counts
    overlapping / small-object / small-hole reasons using the same thresholds as manual postprocess.
    """

    counts: dict[str, int] = {}
    order: list[str] = []

    def _bump(reason: str) -> None:
        if reason not in counts:
            order.append(reason)
            counts[reason] = 0
        counts[reason] += 1

    if settings is None:
        for polygon in polygons:
            reason = polygon.description_invalid_reason()
            if reason is not None:
                _bump(reason)
    else:
        reasons_by_id = polygons_needing_repair(polygons, settings)
        for polygon in polygons:
            for reason in reasons_by_id.get(polygon.id, ()):
                _bump(reason)
    return [(reason, counts[reason]) for reason in order]


def repair_invalid_polygon_descriptions(
    polygons: list[PolygonData],
    settings: VectorGeometrySettings | None = None,
) -> list[PolygonData]:
    """Repair rings that need cleanup.

    Without ``settings``: rebuild keyholes / self-intersections only (legacy topology path).
    With ``settings``: also merge overlapping root families, dissolve small holes, and drop
    small outers using the same thresholds as manual vector postprocess.

    CIF keyhole ``repeated_vertex`` rings are not auto-offered as invalid, but explicit
    repair still splits them when this function is invoked.
    """

    if settings is None:
        if not any(_polygon_ring_needs_explicit_topology_repair(polygon) for polygon in polygons):
            return [polygon.clone() for polygon in polygons]
        return dissolve_self_intersecting_polygons(polygons, include_keyhole_bridges=True)

    needs = polygons_needing_repair(polygons, settings)
    has_keyhole = any(_polygon_ring_has_keyhole_bridge(polygon) for polygon in polygons)
    if not needs and not has_keyhole:
        return [polygon.clone() for polygon in polygons]

    all_reasons = {reason for reasons in needs.values() for reason in reasons}
    work = [polygon.clone() for polygon in polygons]

    if has_keyhole or any(reason in {"repeated_vertex", "self_intersecting"} for reason in all_reasons):
        work = dissolve_self_intersecting_polygons(work, include_keyhole_bridges=True)

    if "overlapping" in all_reasons:
        work = merge_overlapping_root_families(work)
        # Raster merge of adjacent healed rings can recreate keyhole descriptions; heal again.
        work = dissolve_self_intersecting_polygons(work, include_keyhole_bridges=True)

    work = dissolve_small_holes(work, settings.min_hole_area_to_remove_px2)
    work = drop_small_outer_polygons(work, settings.min_outer_area_px2)
    work = drop_orphan_holes(work)
    return collapse_redundant_vertices_in_polygons(work)


def _polygon_topo_points_key(points: list[tuple[float, float]]) -> object:
    n = len(points)
    if n <= TOPOLOGY_CHECK_MAX_VERTICES:
        return tuple((round(float(x_coord), 4), round(float(y_coord), 4)) for x_coord, y_coord in points)
    centroid_x = 0.0
    centroid_y = 0.0
    for x_coord, y_coord in points:
        centroid_x += float(x_coord)
        centroid_y += float(y_coord)
    inv_n = 1.0 / float(n)
    return (
        "dense",
        n,
        int(round(centroid_x * inv_n * 10_000.0)),
        int(round(centroid_y * inv_n * 10_000.0)),
    )


def _single_polygon_topo_signature(polygon: PolygonData) -> tuple[object, ...]:
    return (
        polygon.id,
        bool(polygon.is_hole),
        polygon.parent_id,
        str(polygon.category),
        str(polygon.shape_hint),
        _polygon_topo_points_key(polygon.points),
    )


def _polygons_topo_signature(polygons: list[PolygonData]) -> tuple[tuple[object, ...], ...]:
    return tuple(_single_polygon_topo_signature(polygon) for polygon in sorted(polygons, key=lambda q: q.id))


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


def postprocess_changed_polygon_edit(
    polygons: list[PolygonData],
    settings: VectorGeometrySettings,
    *,
    polygon_id: int | None,
) -> tuple[list[PolygonData], bool, bool]:
    """Apply local postprocess to one edited polygon or reject the edit unchanged.

    Returns ``(polygons, accepted, changed)``.
    """

    if polygon_id is None:
        return [polygon.clone() for polygon in polygons], True, False

    original_trial = next((polygon for polygon in polygons if polygon.id == polygon_id), None)
    if original_trial is None:
        return [polygon.clone() for polygon in polygons], True, False

    before_signature = _single_polygon_topo_signature(original_trial)
    before_layer_signature = _polygons_topo_signature(polygons)
    work = list(polygons)
    work = apply_topology_repair_after_edit(work, changed_polygon_ids={polygon_id})
    trial = next((polygon for polygon in work if polygon.id == polygon_id), None)
    if trial is None:
        work = drop_orphan_holes(work)
        if not work or any(not _ring_has_usable_geometry(polygon.points) for polygon in work):
            return [polygon.clone() for polygon in polygons], False, False
        changed = before_layer_signature != _polygons_topo_signature(work)
        return work, True, changed

    replacement = _postprocess_scoped_single_polygon(trial, settings)
    if replacement is None or len(replacement.points) < len(trial.points):
        return [polygon.clone() for polygon in polygons], False, False

    replacement.id = trial.id
    replacement.parent_id = trial.parent_id
    for index, polygon in enumerate(work):
        if polygon.id == polygon_id:
            work[index] = replacement
            break
    work = drop_orphan_holes(work)
    changed = before_signature != _single_polygon_topo_signature(replacement)
    return work, True, changed


def postprocess_changed_polygon_only(
    polygons: list[PolygonData],
    settings: VectorGeometrySettings,
    *,
    polygon_id: int | None,
) -> tuple[list[PolygonData], bool]:
    """Fast path: apply geometry checks only to the edited polygon."""

    if polygon_id is None:
        return [p.clone() for p in polygons], False
    before_by_id = {p.id: p for p in polygons}
    before_target = before_by_id.get(polygon_id)
    if before_target is None:
        return [p.clone() for p in polygons], False
    before_signature = _single_polygon_topo_signature(before_target)
    work = [p.clone() for p in polygons]
    by_id = {p.id: p for p in work}
    target = by_id.get(polygon_id)
    if target is None:
        return work, False

    replacement = _postprocess_scoped_single_polygon(target, settings)
    if replacement is None:
        return [p.clone() for p in polygons], False

    replacement.id = target.id
    replacement.parent_id = target.parent_id
    for index, poly in enumerate(work):
        if poly.id == polygon_id:
            work[index] = replacement
            break
    work = drop_orphan_holes(work)
    changed = before_signature != _single_polygon_topo_signature(replacement)
    return work, changed


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


def postprocess_vertex_move_edit(
    polygons: list[PolygonData],
    settings: VectorGeometrySettings,
    *,
    polygon_id: int,
    vertex_index: int,
    new_point: tuple[float, float],
) -> tuple[list[PolygonData], bool, int | None]:
    """Fast vertex-move postprocess with structural sharing and scoped collapse."""

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
            if before_merge != _polygons_topo_signature(work):
                if polygon_id not in {polygon.id for polygon in work}:
                    return work, True, resolve_focus_id_after_geometry_pass(polygons, polygon_id, work)
        return work, True, polygon_id

    work = drop_orphan_holes(work)
    return work, True, resolve_focus_id_after_geometry_pass(polygons, polygon_id, work)


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
