"""serializers: split linked polygon rings."""
from __future__ import annotations
from contour.domain import compute_polygon_metrics

from contour.serializers_parts.boundary_vertices_by_distance import (
    _dedupe_closed_points,
    _has_duplicate_points,
    _is_simple_extracted_hole_ring,
)

from contour.serializers_parts.cif_paint_mask_ring import (
    _collapse_recovered_outline,
    _remove_out_and_back_spikes,
)

def _split_linked_polygon_rings(
    points: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    open_points = _dedupe_closed_points(points)
    if not _has_duplicate_points(open_points):
        return open_points, []
    point_count = len(open_points)
    bridge_return_indices: dict[tuple[tuple[float, float], tuple[float, float]], list[int]] = {}
    for return_index in range(1, point_count - 1):
        bridge_return_indices.setdefault(
            (open_points[return_index], open_points[return_index + 1]),
            [],
        ).append(return_index)
    best_span: int | None = None
    best_hole: list[tuple[float, float]] | None = None
    best_outer: list[tuple[float, float]] | None = None
    for start_index in range(0, point_count - 3):
        outer_anchor = open_points[start_index]
        hole_anchor = open_points[start_index + 1]
        if outer_anchor == hole_anchor:
            continue
        for return_index in bridge_return_indices.get((hole_anchor, outer_anchor), ()):
            if return_index < start_index + 3:
                continue
            span = return_index - start_index
            if best_span is not None and span >= best_span:
                continue
            # Spike-clean the hole so A->B->A bridge leftovers do not look like
            # nested keyholes. Do not collinear-collapse the outer yet: that can
            # delete still-needed bridge anchors (e.g. right-edge multi-hole CIF).
            hole_ring = _remove_out_and_back_spikes(
                _dedupe_closed_points(open_points[start_index + 1 : return_index + 1])
            )
            outer_ring = _dedupe_closed_points(
                [outer_anchor, *open_points[return_index + 2 :], *open_points[: start_index + 1]]
            )
            if len(hole_ring) < 3 or len(outer_ring) < 3:
                continue
            hole_area, _, _ = compute_polygon_metrics(hole_ring)
            outer_area, _, _ = compute_polygon_metrics(outer_ring)
            if hole_area > outer_area:
                hole_ring, outer_ring = outer_ring, hole_ring
            if not _is_simple_extracted_hole_ring(hole_ring):
                continue
            best_span = span
            best_hole = hole_ring
            best_outer = outer_ring
    if best_hole is None or best_outer is None:
        return open_points, []
    base_outer, base_holes = _split_linked_polygon_rings(best_outer)
    return _collapse_recovered_outline(base_outer), [
        _collapse_recovered_outline(hole) for hole in (best_hole, *base_holes)
    ]
