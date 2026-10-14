"""editor_scene: edge at."""
from __future__ import annotations
from math import hypot
from PyQt6.QtCore import QPointF
from contour.graphics.geometry import (
    _distance_to_segment,
)
from contour.graphics.tool_mode_logic import (
    is_via_polygon as _is_via_polygon,
)


class EdgeAtMixin:
    def edge_at(
        self,
        scene_pos: QPointF,
        tolerance: float,
        *,
        vertex_tolerance: float | None = None,
    ) -> tuple[int, int] | None:
        vertex_tol = vertex_tolerance if vertex_tolerance is not None else tolerance
        target_x = float(scene_pos.x())
        target_y = float(scene_pos.y())
        best_hit: tuple[int, int] | None = None
        best_distance = float(tolerance)
        for polygon_id in self._move_target_candidate_polygon_ids(scene_pos, padding=tolerance):
            polygon = self._polygons.get(polygon_id)
            if polygon is None or len(polygon.points) < 2:
                continue
            if not self._point_near_polygon_bbox(polygon, target_x, target_y, tolerance):
                continue
            for index, start in enumerate(polygon.points):
                end = polygon.points[(index + 1) % len(polygon.points)]
                if hypot(start[0] - end[0], start[1] - end[1]) < 1e-6:
                    continue
                distance = _distance_to_segment((target_x, target_y), start, end)
                if distance > tolerance:
                    continue
                if hypot(target_x - start[0], target_y - start[1]) <= vertex_tol:
                    continue
                if hypot(target_x - end[0], target_y - end[1]) <= vertex_tol:
                    continue
                if distance < best_distance:
                    best_distance = distance
                    best_hit = (polygon_id, index)
        return best_hit

    def nearest_edge_in_polygon(self, polygon_id: int, scene_pos: QPointF) -> tuple[int, int] | None:
        polygon = self._polygons.get(polygon_id)
        if polygon is None or len(polygon.points) < 2:
            return None
        target_x = float(scene_pos.x())
        target_y = float(scene_pos.y())
        best_index: int | None = None
        best_distance = float("inf")
        for index, start in enumerate(polygon.points):
            end = polygon.points[(index + 1) % len(polygon.points)]
            if hypot(start[0] - end[0], start[1] - end[1]) < 1e-6:
                continue
            distance = _distance_to_segment((target_x, target_y), start, end)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is None:
            return None
        return polygon_id, best_index

    def nearest_edge(self, scene_pos: QPointF, *, max_distance: float | None = None) -> tuple[int, int] | None:
        best_hit: tuple[int, int] | None = None
        best_distance = float("inf") if max_distance is None else float(max_distance)
        sx, sy = scene_pos.x(), scene_pos.y()
        search_limit = best_distance
        candidate_ids = self._move_target_candidate_polygon_ids(
            scene_pos,
            padding=search_limit if max_distance is not None else 0.0,
        ) if max_distance is not None else self._move_target_candidate_polygon_ids()
        for polygon_id in candidate_ids:
            polygon = self._polygons.get(polygon_id)
            if polygon is None:
                continue
            if not self._point_near_polygon_bbox(polygon, sx, sy, search_limit):
                continue
            hit = self.nearest_edge_in_polygon(polygon_id, scene_pos)
            if hit is None:
                continue
            polygon = self._polygons.get(polygon_id)
            if polygon is None:
                continue
            start = polygon.points[hit[1]]
            end = polygon.points[(hit[1] + 1) % len(polygon.points)]
            distance = _distance_to_segment((sx, sy), start, end)
            if distance < best_distance:
                best_distance = distance
                best_hit = hit
        return best_hit

    def delete_polygon_at(self, scene_pos: QPointF) -> bool:
        polygon_id = self.polygon_at(scene_pos)
        if polygon_id is None:
            return False
        return self.delete_polygon(polygon_id)

    def delete_via_at(self, scene_pos: QPointF) -> bool:
        polygon_id = self.polygon_at(scene_pos)
        polygon = self._polygons.get(polygon_id) if polygon_id is not None else None
        if polygon is None or not _is_via_polygon(polygon) or not self.polygon_is_deletable(polygon):
            return False
        self.delete_polygon(polygon_id)
        return True
