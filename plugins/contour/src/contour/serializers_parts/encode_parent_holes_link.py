"""serializers: encode parent holes link."""
from __future__ import annotations
from time import perf_counter
from contour.domain import PolygonData

from contour.serializers_parts.boundary_vertices_by_distance import _rightmost_point_index

from contour.serializers_parts.cv_object_polygon import (
    _note_cif_phase_timing,
    _note_cif_slit_target,
)

from contour.serializers_parts.select_vertex_slit_hole import (
    _edge_parameter,
    _ring_with_orientation,
    _rotate_open_ring,
    _select_vertex_slit_for_hole,
    _trace_linked_hole_cycle,
)

def _encode_parent_with_holes_link_path(parent: PolygonData, holes: list[PolygonData]) -> list[tuple[float, float]]:
    encode_started_at = perf_counter()
    # Image Y grows downwards and CIF Y grows upwards. A positive image-space
    # winding therefore becomes clockwise in CIF; holes use the opposite winding.
    outer = _ring_with_orientation(parent.points, positive_area=True)
    if len(outer) < 3:
        return parent.points

    encoded_holes = [
        (hole, _ring_with_orientation(hole.points, positive_area=False))
        for hole in holes
    ]
    encoded_holes = [(hole, points) for hole, points in encoded_holes if len(points) >= 3]
    encoded_holes.sort(
        key=lambda item: -float(item[1][_rightmost_point_index(item[1])][0]),
    )

    from shapely import make_valid
    from shapely.geometry import Polygon as ShapelyPolygon

    parent_solid = make_valid(
        ShapelyPolygon(
            outer,
            holes=[points for _hole, points in encoded_holes],
        )
    )
    detours_by_ring: dict[int, dict[int, list[_HoleLinkDetour]]] = {}
    linked_hole_indices: set[int] = set()
    for hole_index, (hole, hole_points) in enumerate(encoded_holes):
        slit_started_at = perf_counter()
        linked_hole_rings = [
            (linked_index, encoded_holes[linked_index][1])
            for linked_index in sorted(linked_hole_indices)
        ]
        selected = _select_vertex_slit_for_hole(
            hole_points,
            parent_solid,
            outer=outer,
            linked_hole_rings=linked_hole_rings,
        )
        _note_cif_phase_timing("cif_hole_vertex_slit_search", slit_started_at)
        if selected is None:
            raise ValueError(f"Cannot encode CIF hole {hole.id}: hole is disconnected from parent {parent.id}")
        target_ring_key, target_anchor, insert_index, anchor_index, slit_path = selected
        _note_cif_slit_target(neighbor_hole=target_ring_key >= 0)
        target_ring = outer if target_ring_key < 0 else encoded_holes[target_ring_key][1]
        edge_index = (insert_index - 1) % len(target_ring)
        edge_start = target_ring[edge_index]
        edge_end = target_ring[(edge_index + 1) % len(target_ring)]
        parameter = _edge_parameter(edge_start, edge_end, target_anchor)
        detours_by_ring.setdefault(target_ring_key, {}).setdefault(edge_index, []).append(
            (parameter, target_anchor, slit_path, hole_index, anchor_index)
        )
        linked_hole_indices.add(hole_index)

    path = [outer[0]]
    canonical_start_index: int | None = None
    for edge_index, _edge_start in enumerate(outer):
        detours = sorted(
            detours_by_ring.get(-1, {}).get(edge_index, ()),
            key=lambda item: item[0],
        )
        for _parameter, outer_anchor, slit_path, child_index, child_vertex_index in detours:
            if path[-1] != outer_anchor:
                path.append(outer_anchor)
            child_points = encoded_holes[child_index][1]
            path.extend(slit_path[1:])
            _trace_linked_hole_cycle(
                path,
                child_points,
                hole_index=child_index,
                start_vertex_index=child_vertex_index,
                detours_by_ring=detours_by_ring,
                encoded_holes=encoded_holes,
            )
            path.append(child_points[child_vertex_index])
            path.extend(reversed(slit_path[:-1]))
            if canonical_start_index is None:
                canonical_start_index = len(path) - 1
        edge_end = outer[(edge_index + 1) % len(outer)]
        if path[-1] != edge_end:
            path.append(edge_end)
    if canonical_start_index is not None:
        open_path = path[:-1] if len(path) >= 2 and path[0] == path[-1] else path
        path = _rotate_open_ring(open_path, canonical_start_index)
    _note_cif_phase_timing("cif_hole_link_encode", encode_started_at)
    return path
