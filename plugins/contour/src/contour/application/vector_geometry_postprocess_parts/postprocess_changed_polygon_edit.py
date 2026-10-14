"""vector_geometry_postprocess: postprocess changed polygon edit."""
from __future__ import annotations
from math import hypot
from contour.domain import PolygonData

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _refresh_metrics,
    _ring_has_usable_geometry,
    drop_orphan_holes,
)

from contour.application.vector_geometry_postprocess_parts.merge_candidate_roots_after import apply_topology_repair_after_edit

from contour.application.vector_geometry_postprocess_parts.postprocess_scoped_single_polygon import _postprocess_scoped_single_polygon

from contour.application.vector_geometry_postprocess_parts.repair_invalid_polygon_descriptions import (
    _polygons_topo_signature,
    _single_polygon_topo_signature,
)

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
    trial_has_duplicate_closure = (
        len(trial.points) > 3
        and hypot(
            float(trial.points[0][0]) - float(trial.points[-1][0]),
            float(trial.points[0][1]) - float(trial.points[-1][1]),
        )
        < 1e-5
    )
    removed_only_duplicate_closure = (
        replacement is not None
        and trial_has_duplicate_closure
        and len(replacement.points) + 1 == len(trial.points)
    )
    if replacement is None or (
        len(replacement.points) < len(trial.points) and not removed_only_duplicate_closure
    ):
        return [polygon.clone() for polygon in polygons], False, False

    if removed_only_duplicate_closure:
        replacement.points = [*replacement.points, replacement.points[0]]
        _refresh_metrics(replacement)

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
