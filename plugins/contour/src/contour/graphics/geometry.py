"""Pure geometry helpers used by the polygon editor."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator
from math import atan2, cos, hypot, pi, sin

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QColor, QPainterPath

from ..domain import PolygonData, compute_polygon_metrics
from ..domain.polygon_ring import (
    is_valid_closed_polygon_ring as is_valid_closed_polygon_ring,
)
from ..domain.polygon_ring import (
    is_valid_open_polyline_last_edge as is_valid_open_polyline_last_edge,
)


def _distance_to_segment(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return hypot(px - x1, py - y1)
    t_value = ((px - x1) * dx + (py - y1) * dy) / float(dx * dx + dy * dy)
    t_value = max(0.0, min(1.0, t_value))
    proj_x = x1 + t_value * dx
    proj_y = y1 + t_value * dy
    return hypot(px - proj_x, py - proj_y)


def _points_different(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return hypot(first[0] - second[0], first[1] - second[1]) > 1e-6


def _polygon_points_different(first: list[tuple[float, float]], second: list[tuple[float, float]]) -> bool:
    if len(first) != len(second):
        return True
    return any(_points_different(p0, p1) for p0, p1 in zip(first, second, strict=False))


def _polygon_data_rect(polygon: PolygonData) -> QRectF:
    if polygon.points:
        x_values = [point[0] for point in polygon.points]
        y_values = [point[1] for point in polygon.points]
        return QRectF(
            min(x_values),
            min(y_values),
            max(x_values) - min(x_values),
            max(y_values) - min(y_values),
        ).normalized()
    x_coord, y_coord, width, height = polygon.bbox
    return QRectF(float(x_coord), float(y_coord), float(width), float(height)).normalized()


def _polygons_center(polygons: list[PolygonData]) -> QPointF:
    if not polygons:
        return QPointF(0.0, 0.0)
    boxes = [polygon.bbox for polygon in polygons]
    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    x_max = max(box[0] + box[2] for box in boxes)
    y_max = max(box[1] + box[3] for box in boxes)
    return QPointF((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)


def _path_for_polygon(polygon: PolygonData) -> QPainterPath:
    path = QPainterPath()
    if not polygon.points:
        return path
    if polygon.shape_hint == "box" or polygon.category == "via":
        x_values = [point[0] for point in polygon.points]
        y_values = [point[1] for point in polygon.points]
        path.addEllipse(
            QRectF(min(x_values), min(y_values), max(x_values) - min(x_values), max(y_values) - min(y_values))
        )
        return path
    path.moveTo(polygon.points[0][0], polygon.points[0][1])
    for x_coord, y_coord in polygon.points[1:]:
        path.lineTo(x_coord, y_coord)
    if len(polygon.points) > 2:
        path.closeSubpath()
    return path


_CONTRAST_OBJECT_PALETTE = (
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#469990",
    "#dcbeff",
    "#9a6324",
    "#ffe119",
)


def _bbox_distance_squared(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_left, first_top, first_width, first_height = first
    second_left, second_top, second_width, second_height = second
    first_right = first_left + first_width
    first_bottom = first_top + first_height
    second_right = second_left + second_width
    second_bottom = second_top + second_height
    dx = max(float(first_left - second_right), float(second_left - first_right), 0.0)
    dy = max(float(first_top - second_bottom), float(second_top - first_bottom), 0.0)
    return dx * dx + dy * dy


def _color_distance_squared(first: str, second: str) -> int:
    first_color = QColor(first)
    second_color = QColor(second)
    return sum(
        (first_channel - second_channel) ** 2
        for first_channel, second_channel in zip(
            first_color.getRgb()[:3],
            second_color.getRgb()[:3],
            strict=True,
        )
    )


_PALETTE_RGB_DISTANCES: dict[tuple[str, str], int] = {
    (first, second): _color_distance_squared(first, second)
    for first in _CONTRAST_OBJECT_PALETTE
    for second in _CONTRAST_OBJECT_PALETTE
}
_PALETTE_INDICES = {color: index for index, color in enumerate(_CONTRAST_OBJECT_PALETTE)}


def _grid_cells_for_bbox(
    bbox: tuple[int, int, int, int],
    cell_size: float,
    *,
    margin: float = 0.0,
) -> Iterator[tuple[int, int]]:
    left, top, width, height = bbox
    right = float(left) + float(width)
    bottom = float(top) + float(height)
    x0 = int(math.floor((float(left) - margin) / cell_size))
    x1 = int(math.floor((right + margin) / cell_size))
    y0 = int(math.floor((float(top) - margin) / cell_size))
    y1 = int(math.floor((bottom + margin) / cell_size))
    for cx in range(x0, x1 + 1):
        for cy in range(y0, y1 + 1):
            yield cx, cy


def _contrasting_object_colors(
    polygons: Iterable[PolygonData],
    frame_size: tuple[float, float],
) -> dict[int, str]:
    """Assign deterministic colors that contrast between nearby conductors."""

    polygon_list = list(polygons)
    conductors = {
        polygon.id: polygon
        for polygon in polygon_list
        if not polygon.is_hole and str(polygon.category or "") == "conductor"
    }
    if not conductors:
        return {}

    frame_width, frame_height = frame_size
    proximity = max(12.0, min(64.0, hypot(float(frame_width), float(frame_height)) * 0.01))
    proximity_squared = proximity * proximity
    neighbors = {polygon_id: set() for polygon_id in conductors}
    cell_size = proximity
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for polygon_id, polygon in conductors.items():
        for cell in _grid_cells_for_bbox(polygon.bbox, cell_size, margin=proximity):
            grid[cell].append(polygon_id)
    for polygon_id, polygon in conductors.items():
        candidate_ids: set[int] = set()
        for cell in _grid_cells_for_bbox(polygon.bbox, cell_size, margin=proximity):
            candidate_ids.update(grid[cell])
        for other_id in candidate_ids:
            if other_id == polygon_id:
                continue
            if _bbox_distance_squared(polygon.bbox, conductors[other_id].bbox) > proximity_squared:
                continue
            neighbors[polygon_id].add(other_id)
            neighbors[other_id].add(polygon_id)

    palette_rgb_distances = _PALETTE_RGB_DISTANCES
    palette_indices = _PALETTE_INDICES
    root_colors: dict[int, str] = {}
    coloring_order = sorted(conductors, key=lambda polygon_id: (-len(neighbors[polygon_id]), polygon_id))
    for polygon_id in coloring_order:
        neighbor_colors = [
            root_colors[neighbor_id]
            for neighbor_id in neighbors[polygon_id]
            if neighbor_id in root_colors
        ]
        preferred_index = (int(polygon_id) * 7) % len(_CONTRAST_OBJECT_PALETTE)
        if not neighbor_colors:
            root_colors[polygon_id] = _CONTRAST_OBJECT_PALETTE[preferred_index]
            continue
        root_colors[polygon_id] = max(
            _CONTRAST_OBJECT_PALETTE,
            key=lambda color: (
                min(palette_rgb_distances[color, neighbor_color] for neighbor_color in neighbor_colors),
                -((palette_indices[color] - preferred_index) % len(_CONTRAST_OBJECT_PALETTE)),
            ),
        )

    result = dict(root_colors)
    for polygon in polygon_list:
        if polygon.is_hole and polygon.parent_id in root_colors:
            result[polygon.id] = root_colors[polygon.parent_id]
    return result


def _stable_layer_color(layer_index: int) -> str:
    hue = (45 + int(layer_index) * 97) % 360
    color = QColor()
    color.setHsv(hue, 170, 255)
    return color.name()


def _snap_to_45(start: QPointF, target: QPointF) -> QPointF:
    dx = target.x() - start.x()
    dy = target.y() - start.y()
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return QPointF(target)
    angle = atan2(dy, dx)
    snapped_angle = round(angle / (pi / 4.0)) * (pi / 4.0)
    distance = hypot(dx, dy)
    return QPointF(start.x() + cos(snapped_angle) * distance, start.y() + sin(snapped_angle) * distance)


def _centered_rect(center: QPointF, width: float, height: float) -> QRectF:
    safe_width = max(1.0, float(width))
    safe_height = max(1.0, float(height))
    return QRectF(
        center.x() - safe_width / 2.0,
        center.y() - safe_height / 2.0,
        safe_width,
        safe_height,
    )


def _measurement_label_position(start: QPointF, end: QPointF) -> QPointF:
    dx = end.x() - start.x()
    dy = end.y() - start.y()
    midpoint = QPointF((start.x() + end.x()) / 2.0, (start.y() + end.y()) / 2.0)
    distance = hypot(dx, dy)
    if distance < 1e-6:
        return QPointF(midpoint.x() + 6.0, midpoint.y() - 16.0)
    normal_x = -dy / distance
    normal_y = dx / distance
    if normal_y > 0:
        normal_x *= -1.0
        normal_y *= -1.0
    return QPointF(midpoint.x() + normal_x * 14.0, midpoint.y() + normal_y * 14.0)


def _bbox_from_points(points: list[tuple[float, float]], padding: int = 0) -> tuple[int, int, int, int]:
    array = np.asarray(points, dtype=np.float32)
    x_min = int(np.floor(array[:, 0].min())) - padding
    y_min = int(np.floor(array[:, 1].min())) - padding
    x_max = int(np.ceil(array[:, 0].max())) + padding
    y_max = int(np.ceil(array[:, 1].max())) + padding
    return x_min, y_min, max(1, x_max - x_min + 1), max(1, y_max - y_min + 1)


def _union_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    x_max = max(box[0] + box[2] for box in boxes)
    y_max = max(box[1] + box[3] for box in boxes)
    return x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min)


def _bboxes_intersect(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    return not (
        first[0] + first[2] <= second[0]
        or second[0] + second[2] <= first[0]
        or first[1] + first[3] <= second[1]
        or second[1] + second[3] <= first[1]
    )


def _polygon_reference_point(polygon: PolygonData) -> tuple[float, float]:
    x_coord, y_coord, width, height = polygon.bbox
    return float(x_coord) + float(width) / 2.0, float(y_coord) + float(height) / 2.0


def _polygon_contains_point(polygon: PolygonData, point: tuple[float, float]) -> bool:
    contour = np.asarray(
        [[float(x_coord), float(y_coord)] for x_coord, y_coord in polygon.points],
        dtype=np.float32,
    )
    if contour.shape[0] < 3:
        return False
    return cv2.pointPolygonTest(contour.reshape((-1, 1, 2)), point, False) >= 0.0


def _smallest_containing_polygon(polygon: PolygonData, candidates: list[PolygonData]) -> PolygonData | None:
    point = _polygon_reference_point(polygon)
    containing = [
        candidate
        for candidate in candidates
        if candidate.id != polygon.id and _polygon_contains_point(candidate, point)
    ]
    if not containing:
        return None
    return min(containing, key=lambda candidate: candidate.area)


def resolve_hover_polygon_id(
    polygons_by_id: dict[int, PolygonData],
    hole_children_by_parent: dict[int, list[PolygonData]],
    scene_x: float,
    scene_y: float,
    *,
    contours_by_id: dict[int, np.ndarray] | None = None,
) -> int | None:
    """Pick the smallest polygon under the pointer, preferring holes over outer metal."""
    point = (float(scene_x), float(scene_y))

    def contains(polygon: PolygonData) -> bool:
        contour = None if contours_by_id is None else contours_by_id.get(polygon.id)
        if contour is None:
            return _polygon_contains_point(polygon, point)
        if contour.shape[0] < 3:
            return False
        return cv2.pointPolygonTest(contour, point, False) >= 0.0

    candidates: list[int] = []
    hole_ids_at_point: list[int] = []
    for polygon_id, polygon in polygons_by_id.items():
        if not contains(polygon):
            continue
        if polygon.is_hole:
            hole_ids_at_point.append(polygon_id)
            continue
        hole_children = hole_children_by_parent.get(polygon_id, [])
        if any(contains(hole) for hole in hole_children):
            continue
        candidates.append(polygon_id)
    candidates.extend(hole_ids_at_point)
    if not candidates:
        return None
    return min(candidates, key=lambda polygon_id: float(polygons_by_id[polygon_id].area))


def resolve_conductor_hover_target_id(
    polygons_by_id: dict[int, PolygonData], hovered_polygon_id: int | None
) -> int | None:
    """Return the polygon id whose outline should glow on hover."""
    if hovered_polygon_id is None:
        return None
    if hovered_polygon_id not in polygons_by_id:
        return None
    return hovered_polygon_id


def _clip_bbox_to_scene(bbox: tuple[int, int, int, int], scene_rect: QRectF) -> tuple[int, int, int, int]:
    scene_left = int(np.floor(scene_rect.left()))
    scene_top = int(np.floor(scene_rect.top()))
    scene_right = int(np.ceil(scene_rect.right()))
    scene_bottom = int(np.ceil(scene_rect.bottom()))
    x_coord = max(scene_left, bbox[0])
    y_coord = max(scene_top, bbox[1])
    right = min(scene_right, bbox[0] + bbox[2])
    bottom = min(scene_bottom, bbox[1] + bbox[3])
    return x_coord, y_coord, max(1, right - x_coord), max(1, bottom - y_coord)


def _fill_polygon_on_mask(
    mask: np.ndarray, points: list[tuple[float, float]], origin: tuple[int, int], value: int = 255
) -> None:
    shifted = np.asarray(
        [[round(x_coord - origin[0]), round(y_coord - origin[1])] for x_coord, y_coord in points],
        dtype=np.int32,
    )
    if shifted.shape[0] >= 3:
        cv2.fillPoly(mask, [shifted.reshape((-1, 1, 2))], int(value))


def _draw_polygon_outline_on_mask(
    mask: np.ndarray, points: list[tuple[float, float]], origin: tuple[int, int], value: int = 255
) -> None:
    shifted = np.asarray(
        [[round(x_coord - origin[0]), round(y_coord - origin[1])] for x_coord, y_coord in points],
        dtype=np.int32,
    )
    if shifted.shape[0] >= 3:
        cv2.polylines(mask, [shifted.reshape((-1, 1, 2))], True, int(value), thickness=1, lineType=cv2.LINE_8)


def _draw_stroke_on_mask(
    mask: np.ndarray, points: list[tuple[float, float]], origin: tuple[int, int], thickness: float
) -> None:
    shifted = [(round(x_coord - origin[0]), round(y_coord - origin[1])) for x_coord, y_coord in points]
    line_width = max(1, round(thickness))
    radius = max(1, line_width // 2)
    for start, end in itertools.pairwise(shifted):
        cv2.line(mask, start, end, 255, thickness=line_width, lineType=cv2.LINE_8)
    cv2.circle(mask, shifted[0], radius, 255, thickness=-1, lineType=cv2.LINE_8)
    cv2.circle(mask, shifted[-1], radius, 255, thickness=-1, lineType=cv2.LINE_8)


def _polygon_depth_for_render(
    polygon: PolygonData,
    polygons_by_id: dict[int, PolygonData],
    cache: dict[int, int],
) -> int:
    cached = cache.get(polygon.id)
    if cached is not None:
        return cached
    if polygon.parent_id is None or polygon.parent_id not in polygons_by_id:
        cache[polygon.id] = 0
        return 0
    depth = _polygon_depth_for_render(polygons_by_id[polygon.parent_id], polygons_by_id, cache) + 1
    cache[polygon.id] = depth
    return depth


def _render_polygon_collection_on_mask(mask: np.ndarray, polygons: list[PolygonData], origin: tuple[int, int]) -> None:
    from ..serializers import _stamp_cif_paint_ring_on_mask

    cif_paint_parent_ids = {
        polygon.id for polygon in polygons if _uses_cif_paint_parent(polygon)
    }
    polygons_by_id = {polygon.id: polygon for polygon in polygons}
    depth_cache: dict[int, int] = {}
    ordered_polygons = sorted(
        polygons,
        key=lambda polygon: (_polygon_depth_for_render(polygon, polygons_by_id, depth_cache), polygon.id),
    )
    for polygon in ordered_polygons:
        if _uses_cif_paint_parent(polygon):
            _stamp_cif_paint_ring_on_mask(mask, polygon.cif_paint_ring, origin, value=255)
            continue
        depth = _polygon_depth_for_render(polygon, polygons_by_id, depth_cache)
        if polygon.is_hole and polygon.parent_id in cif_paint_parent_ids:
            continue
        if depth % 2:
            _fill_polygon_on_mask(mask, polygon.points, origin, value=0)
            _draw_polygon_outline_on_mask(mask, polygon.points, origin, value=255)
        else:
            _fill_polygon_on_mask(mask, polygon.points, origin, value=255)


def _uses_cif_paint_parent(polygon: PolygonData) -> bool:
    from ..serializers import _polygon_uses_authored_cif_paint_ring

    return _polygon_uses_authored_cif_paint_ring(polygon)


def _contour_depth(contour_index: int, hierarchy: np.ndarray, cache: dict[int, int]) -> int:
    if contour_index in cache:
        return cache[contour_index]
    parent_index = int(hierarchy[contour_index][3])
    if parent_index < 0:
        cache[contour_index] = 0
        return 0
    depth = _contour_depth(parent_index, hierarchy, cache) + 1
    cache[contour_index] = depth
    return depth


def _polygons_from_mask(mask: np.ndarray, origin: tuple[int, int]) -> list[PolygonData]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy_array = hierarchy[0]
    depth_cache: dict[int, int] = {}
    intermediates: list[tuple[int, int, int, list[tuple[float, float]], float, float, tuple[int, int, int, int]]] = []
    for contour_index, contour in enumerate(contours):
        if contour is None or len(contour) < 3:
            continue
        approx = cv2.approxPolyDP(contour, 1.0, True)
        points = [(float(point[0][0] + origin[0]), float(point[0][1] + origin[1])) for point in approx]
        if len(points) < 3:
            continue
        area, perimeter, bbox = compute_polygon_metrics(points)
        if area <= 0.0 or perimeter <= 0.0:
            continue
        parent_index = int(hierarchy_array[contour_index][3])
        depth = _contour_depth(contour_index, hierarchy_array, depth_cache)
        intermediates.append((contour_index, parent_index, depth, points, area, perimeter, bbox))

    contour_id_to_polygon_id = {
        contour_index: polygon_id
        for polygon_id, (contour_index, _parent_index, _depth, _points, _area, _perimeter, _bbox) in enumerate(
            intermediates, start=1
        )
    }
    polygons: list[PolygonData] = []
    for polygon_id, (contour_index, parent_index, depth, points, area, perimeter, bbox) in enumerate(
        intermediates, start=1
    ):
        del contour_index
        polygons.append(
            PolygonData(
                id=polygon_id,
                points=points,
                is_hole=bool(depth % 2),
                parent_id=None if parent_index < 0 else contour_id_to_polygon_id.get(parent_index),
                area=area,
                perimeter=perimeter,
                bbox=bbox,
            )
        )
    return polygons
