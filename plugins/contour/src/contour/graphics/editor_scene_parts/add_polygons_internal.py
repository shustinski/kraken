"""editor_scene: add polygons internal."""
from __future__ import annotations
from math import hypot
import numpy as np
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
)
from contour.domain import PolygonData, integer_point, integer_points
from contour.graphics.geometry import (
    _distance_to_segment,
)


class AddPolygonsInternalMixin:
    def _recycle_polygon_items(self) -> None:
        self._recycled_polygon_cleanup_timer.stop()
        for item in self._polygon_items.values():
            item.setVisible(False)
            self._recycled_polygon_items.append(item)
        self._polygon_items.clear()

    def _start_recycled_polygon_cleanup(self) -> None:
        if self._recycled_polygon_items:
            self._recycled_polygon_cleanup_timer.start()

    def _drain_recycled_polygon_items(self) -> None:
        # Removing a very large previous result synchronously can block the
        # display of a replacement frame for tens of seconds. Dispose of the
        # unused items in small event-loop batches after the new result is ready.
        batch_size = min(32, len(self._recycled_polygon_items))
        for _index in range(batch_size):
            self.removeItem(self._recycled_polygon_items.pop())
        if not self._recycled_polygon_items:
            self._recycled_polygon_cleanup_timer.stop()

    def _remove_polygon_internal(self, polygon_id: int, emit_signal: bool = True, refresh: bool = True) -> None:
        item = self._polygon_items.pop(polygon_id, None)
        self._polygon_spatial_index.remove(polygon_id)
        self._polygon_hover_contours.pop(polygon_id, None)
        self._unindex_polygon_relationship(self._polygons.pop(polygon_id, None))
        if self._hover_conductor_polygon_id == polygon_id:
            self._hover_conductor_polygon_id = None
        if self._vertex_preview_polygon_id == polygon_id:
            self._vertex_preview_polygon_id = None
        if item is not None:
            self.removeItem(item)
        if self._selected_polygon_id == polygon_id:
            self._selected_polygon_id = None
            self.activePolygonChanged.emit(None)
        self._selected_polygon_ids.discard(polygon_id)
        if refresh:
            self._refresh_all_items()
        if emit_signal:
            self.polygonsChanged.emit()

    def _remove_polygons_internal(
        self,
        polygon_ids: list[int],
        *,
        emit_signal: bool = True,
        refresh: bool = True,
    ) -> None:
        for polygon_id in polygon_ids:
            self._remove_polygon_internal(polygon_id, emit_signal=False, refresh=False)
        if refresh:
            self._refresh_all_items()
        if emit_signal:
            self.polygonsChanged.emit()

    def _add_polygons_internal(
        self,
        polygons: list[PolygonData],
        *,
        emit_signal: bool = True,
        refresh: bool = True,
    ) -> None:
        for polygon in polygons:
            self._add_polygon_internal(
                polygon,
                emit_signal=False,
                refresh=False,
                paint=not refresh,
                defer_pick_z_rebuild=True,
            )
        if refresh:
            self._refresh_all_items()
        else:
            self._rebuild_outer_pick_z_ranks()
            changed_color_ids = self._rebuild_object_colors()
            refresh_ids = changed_color_ids | {polygon.id for polygon in polygons}
            self._refresh_polygon_items_by_id(*refresh_ids)
        if emit_signal:
            self.polygonsChanged.emit()

    def _replace_polygon_points_internal(
        self, polygon_id: int, points: list[tuple[float, float]], emit_signal: bool = True
    ) -> None:
        if polygon_id not in self._polygons:
            return
        refresh_ids = set(self._polygon_edit_family_ids(polygon_id))
        root_id = self._polygon_root_id(polygon_id)
        self._unindex_polygon_relationship(self._polygons.get(polygon_id))
        self._polygons[polygon_id] = self._create_polygon_snapshot(polygon_id, points)
        self._polygon_spatial_index.upsert(self._polygons[polygon_id])
        self._polygon_hover_contours[polygon_id] = np.asarray(
            self._polygons[polygon_id].points,
            dtype=np.float32,
        ).reshape((-1, 1, 2))
        if root_id is not None and root_id in self._polygons:
            self._polygons[root_id].cif_paint_ring = []
        self._index_polygon_relationship(self._polygons[polygon_id])
        refresh_ids.update(self._polygon_edit_family_ids(polygon_id))
        refresh_ids.update(self._rebuild_object_colors())
        self._last_geometry_changed_polygon_id = polygon_id
        self._refresh_polygon_items_by_id(*refresh_ids)
        if emit_signal:
            self.polygonsChanged.emit()

    def _set_vertex_internal(
        self,
        polygon_id: int,
        vertex_index: int,
        point: tuple[float, float],
        emit_signal: bool = True,
    ) -> None:
        if polygon_id not in self._polygons:
            return
        points = self.polygon_points(polygon_id)
        if vertex_index < 0 or vertex_index >= len(points):
            return
        new_point = integer_point(point)
        closed_duplicate_endpoint = (
            len(points) > 2 and hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) < 1e-5
        )
        points[vertex_index] = new_point
        if closed_duplicate_endpoint:
            if vertex_index == 0:
                points[-1] = new_point
            elif vertex_index == len(points) - 1:
                points[0] = new_point
        self._replace_polygon_points_internal(polygon_id, points, emit_signal=emit_signal)

    @staticmethod
    def _translate_edge_points(
        points: list[tuple[float, float]],
        edge_index: int,
        delta: tuple[float, float],
    ) -> list[tuple[float, float]]:
        if edge_index < 0 or edge_index >= len(points):
            return list(points)
        moved = [(float(x), float(y)) for x, y in points]
        dx, dy = float(delta[0]), float(delta[1])
        start_idx = edge_index
        end_idx = (edge_index + 1) % len(moved)
        closed_duplicate_endpoint = (
            len(moved) > 2 and hypot(moved[0][0] - moved[-1][0], moved[0][1] - moved[-1][1]) < 1e-5
        )
        moved[start_idx] = integer_point((moved[start_idx][0] + dx, moved[start_idx][1] + dy))
        moved[end_idx] = integer_point((moved[end_idx][0] + dx, moved[end_idx][1] + dy))
        if closed_duplicate_endpoint and (
            start_idx == 0 or end_idx == 0 or start_idx == len(moved) - 1 or end_idx == len(moved) - 1
        ):
            moved[-1] = moved[0]
        return integer_points(moved)

    def _insert_vertex_internal(
        self,
        polygon_id: int,
        insert_index: int,
        point: tuple[float, float],
        emit_signal: bool = True,
    ) -> None:
        points = self.polygon_points(polygon_id)
        insert_at = max(0, min(len(points), insert_index))
        points.insert(insert_at, integer_point(point))
        self._replace_polygon_points_internal(polygon_id, points, emit_signal=emit_signal)

    def _remove_vertex_internal(self, polygon_id: int, vertex_index: int, emit_signal: bool = True) -> None:
        points = self.polygon_points(polygon_id)
        if len(points) <= 3:
            return
        points.pop(vertex_index)
        self._replace_polygon_points_internal(polygon_id, points, emit_signal=emit_signal)

    def _nearest_segment_insert_index(self, polygon_id: int, scene_pos: QPointF) -> int:
        polygon = self._polygons[polygon_id]
        points = polygon.points
        if len(points) < 2:
            return len(points)
        target_x = scene_pos.x()
        target_y = scene_pos.y()
        best_index = 1
        best_distance = float("inf")
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            distance = _distance_to_segment((target_x, target_y), start, end)
            if distance < best_distance:
                best_distance = distance
                best_index = index + 1
        return best_index

    def _set_measurement_marker(self, marker: QGraphicsEllipseItem, point: QPointF) -> None:
        radius = 3.0
        marker.setPos(point)
        marker.setRect(QRectF(-radius, -radius, radius * 2.0, radius * 2.0))
        marker.show()
