"""serializers: boundary vertices by distance."""
from __future__ import annotations
from bisect import bisect_left
import numpy as np
from contour.domain import PolygonData

from contour.serializers_parts.module_state import _BOUNDARY_SNAP_NEIGHBOR_WINDOW

from contour.serializers_parts.cv_object_polygon import _pixel_box_center_and_size

def _polygon_to_cif_line(polygon: PolygonData, image_width: int, image_height: int) -> str:
    if polygon.shape_hint == "box":
        x_values = [point[0] for point in polygon.points]
        y_values = [point[1] for point in polygon.points]
        if len(x_values) < 4 or len(y_values) < 4:
            return ""
        center_x, center_y, width, height = _pixel_box_center_and_size(
            min(x_values),
            min(y_values),
            max(x_values),
            max(y_values),
        )
        cif_x = max(0, min(image_width, center_x))
        cif_y = max(0, min(image_height, round(image_height - center_y)))
        return f"B {width} {height} {cif_x} {cif_y};"
    points = []
    for x_coord, y_coord in polygon.points:
        cif_x = max(0, min(image_width, round(x_coord)))
        cif_y = max(0, min(image_height, round(image_height - y_coord)))
        points.append((cif_x, cif_y))
    if len(points) < 3:
        return ""
    if points[0] != points[-1]:
        points.append(points[0])
    coordinates = " ".join(f"{x_coord} {y_coord}" for x_coord, y_coord in points)
    return f"P {coordinates};"

def _to_local_int_points(points: list[tuple[float, float]], left: int, top: int) -> np.ndarray:
    return np.array(
        [[round(x_coord - left), round(y_coord - top)] for x_coord, y_coord in points],
        dtype=np.int32,
    )

def _dedupe_closed_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    first_x, first_y = points[0]
    if isinstance(first_x, int) and isinstance(first_y, int):
        source: list[tuple[float, float]] = [(int(x_coord), int(y_coord)) for x_coord, y_coord in points]
    else:
        source = [(float(x_coord), float(y_coord)) for x_coord, y_coord in points]
    cleaned: list[tuple[float, float]] = [source[0]]
    for point in source[1:]:
        if point != cleaned[-1]:
            cleaned.append(point)
    while len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned = cleaned[:-1]
    return cleaned

def _ring_has_retracing_entry_edge(ring: list[tuple[float, float]]) -> bool:
    """True when closing the ring immediately retraces its first edge.

    Legacy CIF keyholes sometimes repeat the bridge vertex, so a zero-length
    ``A -> A`` pair is treated as the hole mouth and the real bridge is swallowed
    into the hole outline as a long spike.
    """

    return len(ring) >= 3 and ring[-1] == ring[1]

def _is_simple_extracted_hole_ring(ring: list[tuple[float, float]]) -> bool:
    """Reject nested keyholes that still contain their own bridge vertices."""

    return len(ring) >= 3 and not _ring_has_retracing_entry_edge(ring) and not _has_duplicate_points(ring)

def _has_duplicate_points(points: list[tuple[float, float]]) -> bool:
    if len(points) < 2:
        return False
    seen: set[tuple[float, float]] = set()
    for point in points:
        if point in seen:
            return True
        seen.add(point)
    return False

def _rightmost_point_index(points: list[tuple[float, float]]) -> int:
    return max(range(len(points)), key=lambda index: (points[index][0], -points[index][1]))

def _point_distance_squared(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    delta_x = float(left[0]) - float(right[0])
    delta_y = float(left[1]) - float(right[1])
    return delta_x * delta_x + delta_y * delta_y

def _boundary_vertices_by_distance(
    boundary: list[tuple[float, float]],
    hole_start: tuple[float, float],
    *,
    neighbor_window: int = _BOUNDARY_SNAP_NEIGHBOR_WINDOW,
    prefer_same_x_side: bool = True,
) -> list[tuple[tuple[float, float], int]]:
    """Rank boundary vertices by distance; bisect window first, then the rest."""

    vertex_count = len(boundary)
    if vertex_count < 3:
        return []

    hole_x = float(hole_start[0])
    indexed_boundary = sorted(enumerate(boundary), key=lambda item: float(item[1][0]))
    x_coords = [float(point[0]) for _, point in indexed_boundary]
    anchor_index = bisect_left(x_coords, hole_x)
    window = max(1, int(neighbor_window))
    start = max(0, anchor_index - window)
    end = min(len(indexed_boundary), anchor_index + window + 1)
    window_pairs = [(boundary[index], index) for index, _point in indexed_boundary[start:end]]
    window_keys = {index for index, _point in indexed_boundary[start:end]}

    def _maybe_prioritize_same_x_side(
        pairs: list[tuple[tuple[float, float], int]],
    ) -> list[tuple[tuple[float, float], int]]:
        if not prefer_same_x_side:
            return pairs
        same_x_side = [item for item in pairs if float(item[0][0]) + 1e-9 >= hole_x]
        if same_x_side:
            return same_x_side
        return pairs

    window_ranked = sorted(
        _maybe_prioritize_same_x_side(window_pairs),
        key=lambda item: _point_distance_squared(hole_start, item[0]),
    )
    remaining_pairs = [(boundary[index], index) for index in range(vertex_count) if index not in window_keys]
    remaining_ranked = sorted(
        _maybe_prioritize_same_x_side(remaining_pairs),
        key=lambda item: _point_distance_squared(hole_start, item[0]),
    )
    return window_ranked + remaining_ranked

def _snap_hole_to_boundary_vertex(
    boundary: list[tuple[float, float]],
    hole_start: tuple[float, float],
    *,
    neighbor_window: int = _BOUNDARY_SNAP_NEIGHBOR_WINDOW,
) -> tuple[tuple[float, float], int] | None:
    """Return the nearest boundary vertex candidate (without material coverage check)."""

    ranked = _boundary_vertices_by_distance(
        boundary,
        hole_start,
        neighbor_window=neighbor_window,
    )
    if not ranked:
        return None
    outer_anchor, boundary_index = ranked[0]
    return outer_anchor, boundary_index
