"""editor_scene: move target candidate polygon."""
from __future__ import annotations
from math import hypot
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import (
    QGraphicsItem,
)
from contour.domain import PolygonData
from contour.graphics_items import (
    EditablePolygonItem,
    VertexHandleItem,
)
from contour.graphics.geometry import (
    _distance_to_segment,
)

from contour.graphics.editor_scene_parts.module_state import _PICK_CYCLE_DISTANCE


class MoveTargetCandidatePolygonMixin:
    def _polygon_id_from_item(self, item: QGraphicsItem | None) -> int | None:
        if isinstance(item, VertexHandleItem):
            return item.polygon_id
        if isinstance(item, EditablePolygonItem):
            return item.polygon_id
        if item is not None:
            parent = item.parentItem()
            if isinstance(parent, EditablePolygonItem):
                return parent.polygon_id
        return None

    def _reset_pick_cycle(self) -> None:
        self._pick_cycle_pos = None
        self._pick_cycle_ids = []
        self._pick_cycle_index = 0

    def polygons_at(self, scene_pos: QPointF) -> list[int]:
        ordered: list[int] = []
        seen: set[int] = set()
        for item in self.items(scene_pos):
            polygon_id = self._polygon_id_from_item(item)
            if polygon_id is None or polygon_id in seen:
                continue
            seen.add(polygon_id)
            ordered.append(polygon_id)
        return ordered

    def polygon_at(self, scene_pos: QPointF, *, cycle: bool = False) -> int | None:
        hits = self.polygons_at(scene_pos)
        if not hits:
            self._reset_pick_cycle()
            return None
        if not cycle or len(hits) == 1:
            self._reset_pick_cycle()
            return min(hits, key=lambda polygon_id: float(self._polygons[polygon_id].area))

        hits = sorted(hits, key=lambda polygon_id: float(self._polygons[polygon_id].area))
        same_spot = (
            self._pick_cycle_pos is not None
            and hypot(scene_pos.x() - self._pick_cycle_pos.x(), scene_pos.y() - self._pick_cycle_pos.y())
            <= _PICK_CYCLE_DISTANCE
            and hits == self._pick_cycle_ids
        )
        if same_spot:
            self._pick_cycle_index = (self._pick_cycle_index + 1) % len(hits)
        else:
            self._pick_cycle_pos = QPointF(scene_pos)
            self._pick_cycle_ids = list(hits)
            self._pick_cycle_index = 0
        return hits[self._pick_cycle_index]

    def polygon_at_nearest_edge(self, scene_pos: QPointF, tolerance: float) -> int | None:
        hit = self.polygon_at(scene_pos)
        if hit is not None:
            return hit
        best_id: int | None = None
        best_distance = float(tolerance)
        target_x = float(scene_pos.x())
        target_y = float(scene_pos.y())
        for polygon_id, polygon in self._polygons.items():
            if len(polygon.points) < 2:
                continue
            if not self._point_near_polygon_bbox(polygon, target_x, target_y, tolerance):
                continue
            for index, start in enumerate(polygon.points):
                end = polygon.points[(index + 1) % len(polygon.points)]
                distance = _distance_to_segment((target_x, target_y), start, end)
                if distance < best_distance:
                    best_distance = distance
                    best_id = polygon_id
        return best_id

    @staticmethod
    def _point_near_polygon_bbox(
        polygon: PolygonData,
        x: float,
        y: float,
        padding: float,
    ) -> bool:
        pad = max(0.0, float(padding))
        bbox_x, bbox_y, bbox_w, bbox_h = polygon.bbox
        return (
            bbox_x - pad <= x <= bbox_x + bbox_w + pad
            and bbox_y - pad <= y <= bbox_y + bbox_h + pad
        )

    def _move_target_candidate_polygon_ids(
        self,
        scene_pos: QPointF | None = None,
        *,
        padding: float = 0.0,
    ) -> list[int]:
        nearby_ids = (
            None
            if scene_pos is None
            else self._polygon_spatial_index.query_point(
                scene_pos.x(),
                scene_pos.y(),
                padding=padding,
            )
        )
        candidate_ids: list[int] = []
        if self._selected_polygon_id is not None and (
            nearby_ids is None or self._selected_polygon_id in nearby_ids
        ):
            candidate_ids.append(self._selected_polygon_id)
        candidate_ids.extend(
            polygon_id
            for polygon_id in sorted(self._selected_polygon_ids)
            if polygon_id != self._selected_polygon_id
            and (nearby_ids is None or polygon_id in nearby_ids)
        )
        candidate_ids.extend(
            polygon_id
            for polygon_id in sorted(self._polygons)
            if polygon_id not in self._selected_polygon_ids
            and (nearby_ids is None or polygon_id in nearby_ids)
        )
        return candidate_ids

    def vertex_at(self, scene_pos: QPointF, tolerance: float) -> tuple[int, int] | None:
        target_x = float(scene_pos.x())
        target_y = float(scene_pos.y())
        for polygon_id in self._move_target_candidate_polygon_ids(scene_pos, padding=tolerance):
            polygon = self._polygons.get(polygon_id)
            if polygon is None:
                continue
            if not self._point_near_polygon_bbox(polygon, target_x, target_y, tolerance):
                continue
            for index, (x_coord, y_coord) in enumerate(polygon.points):
                if hypot(target_x - x_coord, target_y - y_coord) <= tolerance:
                    return polygon_id, index
        return None

    def nearest_vertex_in_polygon(self, polygon_id: int, scene_pos: QPointF) -> tuple[int, int] | None:
        polygon = self._polygons.get(polygon_id)
        if polygon is None or not polygon.points:
            return None
        best_index: int | None = None
        best_distance = float("inf")
        sx, sy = scene_pos.x(), scene_pos.y()
        for index, (x_coord, y_coord) in enumerate(polygon.points):
            distance = hypot(sx - x_coord, sy - y_coord)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is None:
            return None
        return polygon_id, best_index

    def nearest_vertex(self, scene_pos: QPointF, *, max_distance: float | None = None) -> tuple[int, int] | None:
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
            for index, (x_coord, y_coord) in enumerate(polygon.points):
                distance = hypot(sx - x_coord, sy - y_coord)
                if distance < best_distance:
                    best_distance = distance
                    best_hit = (polygon_id, index)
        return best_hit
