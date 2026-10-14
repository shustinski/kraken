"""vector_geometry_postprocess: root id map."""
from __future__ import annotations
from contour.domain import PolygonData

from contour.application.vector_geometry_postprocess_parts.apply_spike_removal_polygon import _centroid

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _is_via_like,
    _mask_helpers,
    _polygon_contains_point,
)

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
