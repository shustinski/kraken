"""vector_geometry_postprocess: merge candidate roots near."""
from __future__ import annotations
import cv2
import numpy as np
from contour.domain import PolygonData

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _bboxes_intersect,
    _mask_helpers,
    _root_id_for_polygon,
)

from contour.application.vector_geometry_postprocess_parts.root_id_map import (
    _collect_family,
    _family_bbox,
    _root_family_bboxes,
)

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
    from contour.application.vector_geometry_postprocess_parts.merge_candidate_roots_after import _expand_roots_by_touching_bboxes
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
