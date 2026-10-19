"""serializers: select vertex slit hole."""
from __future__ import annotations

from bisect import bisect_left
from typing import TYPE_CHECKING

import numpy as np

from contour.domain import PolygonData
from contour.serializers_parts.boundary_vertices_by_distance import (
    _dedupe_closed_points,
    _point_distance_squared,
    _rightmost_point_index,
)
from contour.serializers_parts.module_state import _BOUNDARY_SNAP_NEIGHBOR_WINDOW


def _select_vertex_slit_for_hole(
    hole_points: list[tuple[float, float]],
    parent_solid,
    *,
    outer: list[tuple[float, float]],
    linked_hole_rings: list[tuple[int, list[tuple[float, float]]]] | None = None,
) -> tuple[int, tuple[float, float], int, int, list[tuple[float, float]]] | None:
    """Pick the shortest vertex-to-vertex slit that stays inside parent material.

    Targets include the outer ring and any already-linked sibling holes.
    Returns ``(target_ring_key, target_anchor, insert_index, hole_vertex_index, slit_path)``.
    ``target_ring_key`` is ``-1`` for the outer ring, otherwise the encoded-hole index.
    """

    from shapely.geometry import LineString

    targets: list[tuple[int, list[tuple[float, float]]]] = [(-1, outer)]
    if linked_hole_rings:
        targets.extend(linked_hole_rings)

    # Keep the legacy window-first tie order, but sort each ring by X only
    # once. Distances and the current best radius eliminate irrelevant
    # vertices before the expensive Python ranking and geometry predicates.
    indexed_targets = []
    for target_key, ring in targets:
        if len(ring) < 3:
            continue
        coordinates = np.asarray(ring, dtype=np.float64)
        x_order = sorted(range(len(ring)), key=lambda index: float(ring[index][0]))
        indexed_targets.append((target_key, ring, coordinates, x_order, [float(ring[i][0]) for i in x_order]))

    best: tuple[float, int, tuple[float, float], int, int, list[tuple[float, float]]] | None = None
    anchor_index = _rightmost_point_index(hole_points)
    hole_vertex_indices = [anchor_index] + [
        index for index in range(len(hole_points)) if index != anchor_index
    ]
    for hole_index in hole_vertex_indices:
        hole_anchor = hole_points[hole_index]
        for target_key, target_ring, coordinates, x_order, x_coords in indexed_targets:
            dx = coordinates[:, 0] - float(hole_anchor[0])
            dy = coordinates[:, 1] - float(hole_anchor[1])
            distances = dx * dx + dy * dy
            candidates = np.flatnonzero(distances < best[0]) if best is not None else np.arange(len(target_ring))
            if not len(candidates):
                continue
            anchor = bisect_left(x_coords, float(hole_anchor[0]))
            window = max(1, int(_BOUNDARY_SNAP_NEIGHBOR_WINDOW))
            window_order = {index: rank for rank, index in enumerate(x_order[max(0, anchor - window):anchor + window + 1])}
            ranked = sorted(
                map(int, candidates),
                key=lambda index: (
                    float(distances[index]),
                    0 if index in window_order else 1,
                    window_order.get(index, index),
                ),
            )
            for insert_index in ranked:
                target_anchor = target_ring[insert_index]
                distance_squared = _point_distance_squared(hole_anchor, target_anchor)
                if best is not None and distance_squared >= best[0]:
                    break
                if not parent_solid.covers(LineString([target_anchor, hole_anchor])):
                    continue
                best = (
                    distance_squared,
                    target_key,
                    target_anchor,
                    insert_index,
                    hole_index,
                    [target_anchor, hole_anchor],
                )
                break
    if best is None:
        return None
    _distance_squared, target_key, target_anchor, insert_index, hole_index, slit_path = best
    return target_key, target_anchor, insert_index, hole_index, slit_path

def _trace_linked_hole_cycle(
    path: list[tuple[float, float]],
    hole_points: list[tuple[float, float]],
    *,
    hole_index: int,
    start_vertex_index: int,
    detours_by_ring: dict[int, dict[int, list[_HoleLinkDetour]]],
    encoded_holes: list[tuple[PolygonData, list[tuple[float, float]]]],
) -> None:
    """Walk one closed hole ring, expanding nested keyhole detours along the way."""

    vertex_count = len(hole_points)
    for step in range(vertex_count):
        edge_index = (start_vertex_index + step) % vertex_count
        detours = sorted(
            detours_by_ring.get(hole_index, {}).get(edge_index, ()),
            key=lambda item: item[0],
        )
        for _parameter, anchor, slit_path, child_index, child_vertex_index in detours:
            if path[-1] != anchor:
                path.append(anchor)
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
        if step >= vertex_count - 1:
            continue
        next_vertex = hole_points[(edge_index + 1) % vertex_count]
        if path[-1] != next_vertex:
            path.append(next_vertex)

def _rotate_open_ring(points: list[tuple[float, float]], start_index: int) -> list[tuple[float, float]]:
    return points[start_index:] + points[:start_index]

def _signed_ring_area(points: list[tuple[float, float]]) -> float:
    area_twice = 0.0
    for index, (x_coord, y_coord) in enumerate(points):
        next_x, next_y = points[(index + 1) % len(points)]
        area_twice += float(x_coord) * float(next_y) - float(next_x) * float(y_coord)
    return area_twice * 0.5

def _ring_with_orientation(
    points: list[tuple[float, float]],
    *,
    positive_area: bool,
) -> list[tuple[float, float]]:
    ring = _dedupe_closed_points(points)
    if len(ring) < 3:
        return ring
    if (_signed_ring_area(ring) > 0.0) != positive_area:
        ring.reverse()
    return ring

def _edge_parameter(
    start: tuple[float, float],
    end: tuple[float, float],
    point: tuple[float, float],
) -> float:
    delta_x = float(end[0]) - float(start[0])
    delta_y = float(end[1]) - float(start[1])
    if abs(delta_x) >= abs(delta_y) and abs(delta_x) > 1e-12:
        return (float(point[0]) - float(start[0])) / delta_x
    if abs(delta_y) > 1e-12:
        return (float(point[1]) - float(start[1])) / delta_y
    return 0.0

if TYPE_CHECKING:
    from contour.serializers_parts.module_state import _HoleLinkDetour
