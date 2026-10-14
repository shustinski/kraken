"""vector_geometry_postprocess: merge candidate roots after."""
from __future__ import annotations
from contour.domain import PolygonData

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _bboxes_intersect,
    _is_via_like,
    _mask_helpers,
    _root_id_for_polygon,
    collapse_redundant_vertices_for_polygon_ids,
    collapse_redundant_vertices_in_polygons,
    drop_orphan_holes,
    reparent_polygons_orphaned_by_ids,
)

from contour.application.vector_geometry_postprocess_parts.dissolve_self_intersecting_polygons import dissolve_self_intersecting_polygons

from contour.application.vector_geometry_postprocess_parts.merge_overlapping_root_families import (
    merge_overlapping_root_families,
    merge_overlapping_root_families_near_polygons,
)

from contour.application.vector_geometry_postprocess_parts.root_id_map import (
    _collect_family,
    _root_family_bboxes,
)

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
