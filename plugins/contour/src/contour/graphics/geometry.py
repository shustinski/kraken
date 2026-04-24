"""Pure geometry helpers used by the polygon editor."""

from __future__ import annotations

import itertools
from math import atan2, cos, hypot, pi, sin

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QColor, QPainterPath

from ..domain import PolygonData, compute_polygon_metrics


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


def _stable_object_color(polygon_id: int) -> str:
    hue = (int(polygon_id) * 137) % 360
    color = QColor()
    color.setHsv(hue, 190, 245)
    return color.name()


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
    polygons_by_id = {polygon.id: polygon for polygon in polygons}
    depth_cache: dict[int, int] = {}
    ordered_polygons = sorted(
        polygons,
        key=lambda polygon: (_polygon_depth_for_render(polygon, polygons_by_id, depth_cache), polygon.id),
    )
    for polygon in ordered_polygons:
        depth = _polygon_depth_for_render(polygon, polygons_by_id, depth_cache)
        if depth % 2:
            _fill_polygon_on_mask(mask, polygon.points, origin, value=0)
            _draw_polygon_outline_on_mask(mask, polygon.points, origin, value=255)
        else:
            _fill_polygon_on_mask(mask, polygon.points, origin, value=255)


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
