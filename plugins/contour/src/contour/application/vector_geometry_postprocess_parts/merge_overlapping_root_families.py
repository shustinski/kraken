"""vector_geometry_postprocess: merge overlapping root families."""
from __future__ import annotations
import numpy as np
from contour.domain import PolygonData

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _bboxes_intersect,
    _is_via_like,
    _mask_helpers,
    _refresh_metrics,
    drop_orphan_holes,
)

from contour.application.vector_geometry_postprocess_parts.merge_candidate_roots_near import (
    _families_mask_overlap_cached,
    _merge_candidate_roots_near_polygons,
)

from contour.application.vector_geometry_postprocess_parts.root_id_map import (
    _UnionFind,
    _children_by_parent,
    _collect_family,
    _next_polygon_id_after,
    _relabel_polygon_ids,
    _root_family_bboxes,
    _stamp_visual_metadata,
)

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
