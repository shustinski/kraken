"""vector_geometry_postprocess: overlapping root family ids."""
from __future__ import annotations
from contour.domain import PolygonData

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _bboxes_intersect,
    _is_via_like,
    _mask_helpers,
)

from contour.application.vector_geometry_postprocess_parts.merge_candidate_roots_near import _families_mask_overlap_cached

from contour.application.vector_geometry_postprocess_parts.rebuild_family_filled_mask import (
    _polygon_ring_needs_explicit_topology_repair,
    _rebuild_family_from_filled_mask,
)

from contour.application.vector_geometry_postprocess_parts.repair_family_by_keyhole import _repair_family_by_keyhole_split

from contour.application.vector_geometry_postprocess_parts.root_id_map import (
    _children_by_parent,
    _collect_family,
    _pad_bbox,
    _root_family_bboxes,
)

def _dissolve_family_to_simple_rings(family: list[PolygonData]) -> list[PolygonData]:
    from ...graphics.brush_vector import region_geometry, shapely_to_polygon_data_list

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
    _, _, _, _union_bbox = _mask_helpers()
    touch_union = _pad_bbox(_union_bbox(touch_bboxes), 2)
    candidate_roots: set[int] = set()
    for polygon in removed_polygons:
        if polygon.parent_id is None and not polygon.is_hole and not _is_via_like(polygon):
            candidate_roots.add(polygon.id)
    for polygon in polygons:
        if polygon.parent_id is not None or polygon.is_hole or _is_via_like(polygon):
            continue
        if polygon.bbox is None:
            continue
        if _bboxes_intersect(_pad_bbox(polygon.bbox, 2), touch_union):
            candidate_roots.add(polygon.id)
    return candidate_roots

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
