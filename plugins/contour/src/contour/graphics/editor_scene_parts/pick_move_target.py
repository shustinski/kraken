"""editor_scene: pick move target."""
from __future__ import annotations
from math import hypot
from PyQt6.QtCore import QPointF
from contour.graphics.geometry import (
    _distance_to_segment,
)


class PickMoveTargetMixin:
    def pick_move_target(
        self,
        scene_pos: QPointF,
        vertex_tolerance: float,
        edge_tolerance: float,
        *,
        strict: bool = False,
    ) -> tuple[str, int, int] | None:
        vertex_hit = self.vertex_at(scene_pos, vertex_tolerance)
        if vertex_hit is None:
            vertex_hit = self.vertex_at(scene_pos, max(vertex_tolerance, edge_tolerance))
        if vertex_hit is not None:
            return ("vertex", vertex_hit[0], vertex_hit[1])

        edge_hit = self.edge_at(scene_pos, edge_tolerance, vertex_tolerance=vertex_tolerance)
        if edge_hit is not None:
            return ("edge", edge_hit[0], edge_hit[1])

        selected_polygon_id = self._selected_polygon_id
        clicked_polygon_id = self.polygon_at(scene_pos)
        target_polygon_id = selected_polygon_id or clicked_polygon_id
        if target_polygon_id is not None:
            vertex_hit = self.nearest_vertex_in_polygon(target_polygon_id, scene_pos)
            if vertex_hit is not None:
                if not strict:
                    return ("vertex", vertex_hit[0], vertex_hit[1])
                polygon = self._polygons.get(target_polygon_id)
                if polygon is not None:
                    point = polygon.points[vertex_hit[1]]
                    if hypot(scene_pos.x() - point[0], scene_pos.y() - point[1]) <= max(
                        vertex_tolerance, edge_tolerance
                    ):
                        return ("vertex", vertex_hit[0], vertex_hit[1])
            edge_hit = self.nearest_edge_in_polygon(target_polygon_id, scene_pos)
            if edge_hit is not None:
                polygon = self._polygons.get(target_polygon_id)
                if polygon is not None and len(polygon.points) >= 2:
                    start = polygon.points[edge_hit[1]]
                    end = polygon.points[(edge_hit[1] + 1) % len(polygon.points)]
                    distance = _distance_to_segment((scene_pos.x(), scene_pos.y()), start, end)
                    if not strict or distance <= edge_tolerance:
                        return ("edge", edge_hit[0], edge_hit[1])

        vertex_hit = self.nearest_vertex(
            scene_pos,
            max_distance=max(vertex_tolerance, edge_tolerance),
        )
        if vertex_hit is not None:
            if not strict:
                return ("vertex", vertex_hit[0], vertex_hit[1])
            polygon = self._polygons.get(vertex_hit[0])
            if polygon is not None:
                point = polygon.points[vertex_hit[1]]
                if hypot(scene_pos.x() - point[0], scene_pos.y() - point[1]) <= max(vertex_tolerance, edge_tolerance):
                    return ("vertex", vertex_hit[0], vertex_hit[1])
        edge_hit = self.nearest_edge(
            scene_pos,
            max_distance=edge_tolerance,
        )
        if edge_hit is not None:
            polygon = self._polygons.get(edge_hit[0])
            if polygon is not None and len(polygon.points) >= 2:
                start = polygon.points[edge_hit[1]]
                end = polygon.points[(edge_hit[1] + 1) % len(polygon.points)]
                distance = _distance_to_segment((scene_pos.x(), scene_pos.y()), start, end)
                if not strict or distance <= edge_tolerance:
                    return ("edge", edge_hit[0], edge_hit[1])
        return None
