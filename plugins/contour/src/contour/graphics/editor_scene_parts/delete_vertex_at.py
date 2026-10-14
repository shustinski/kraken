"""editor_scene: delete vertex at."""
from __future__ import annotations
from math import hypot
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import (
    QBrush,
    QPainterPath,
    QPen,
)
from ...application.vector_geometry_postprocess import (
    apply_topology_repair_after_edit,
    postprocess_changed_polygon_edit,
    resolve_focus_id_after_geometry_pass,
)
from contour.commands import (
    AddPolygonsCommand,
    AddVertexCommand,
    ReplacePolygonSetCommand,
)
from contour.domain import PolygonData, compute_polygon_metrics, integer_point, integer_points
from ...domain.polygon_ring import collapse_redundant_polyline_vertices, is_valid_closed_polygon_ring
from contour.graphics_items import (
    _closed_polygon_path,
    _display_path_for_polygon,
    _hole_display_hidden,
)
from contour.i18n import tr


class DeleteVertexAtMixin:
    def _add_pasted_polygon_set(
        self,
        polygons: list[PolygonData],
    ) -> list[int]:
        id_map: dict[int, int] = {}
        new_polygons: list[PolygonData] = []
        for polygon in polygons:
            new_id = self._next_polygon_id
            self._next_polygon_id += 1
            id_map[polygon.id] = new_id
            cloned = polygon.clone()
            cloned.id = new_id
            new_polygons.append(cloned)
        for polygon in new_polygons:
            polygon.parent_id = (
                None
                if polygon.parent_id is None
                else id_map.get(polygon.parent_id)
            )
        self.undo_stack.push(
            AddPolygonsCommand(
                self,
                new_polygons,
                "Paste polygons",
                select_after_redo=True,
            )
        )
        return [polygon.id for polygon in new_polygons]

    def add_vertex_at(self, polygon_id: int, scene_pos: QPointF) -> bool:
        if not self.can_add_polygon() or polygon_id not in self._polygons:
            return False
        insert_index = self._nearest_segment_insert_index(polygon_id, scene_pos)
        new_point = integer_point((scene_pos.x(), scene_pos.y()))
        points = self.polygon_points(polygon_id)
        insert_at = max(0, min(len(points), insert_index))
        trial = list(points)
        trial.insert(insert_at, new_point)
        collapsed = collapse_redundant_polyline_vertices(trial, closed=True, min_vertices=3)
        if len(collapsed) <= len(points):
            return False
        if not is_valid_closed_polygon_ring(trial):
            self.warn_invalid_polygon_geometry()
            return False
        self.undo_stack.push(AddVertexCommand(self, polygon_id, insert_index, new_point))
        processed, accepted, _changed = postprocess_changed_polygon_edit(
            self.get_polygons(),
            self._vector_geometry_settings,
            polygon_id=polygon_id,
        )
        if not accepted:
            self.undo_stack.undo()
            self.warn_invalid_polygon_geometry()
            return False
        if _changed:
            before = self.get_polygons()
            self.undo_stack.push(ReplacePolygonSetCommand(self, before, processed, "Vector geometry cleanup"))
        self.select_polygon(polygon_id)
        return True

    def delete_vertex_at(self, scene_pos: QPointF, tolerance: float) -> bool:
        if not self.can_add_polygon():
            return False
        hit = self.vertex_at(scene_pos, tolerance)
        if hit is None:
            return False
        polygon_id, vertex_index = hit
        polygon = self._polygons[polygon_id]
        if len(polygon.points) <= 3:
            self.logRequested.emit(tr("polygon_min_vertices_log", language=self._ui_language))
            return False
        before = self.get_polygons()
        after_delete = [item.clone() for item in before]
        target = next((item for item in after_delete if item.id == polygon_id), None)
        if target is None or vertex_index < 0 or vertex_index >= len(target.points):
            return False
        points = list(target.points)
        points.pop(vertex_index)
        target.points = integer_points(points)
        area, perimeter, bbox = compute_polygon_metrics(target.points)
        target.area = float(area)
        target.perimeter = float(perimeter)
        target.bbox = bbox
        after_delete = apply_topology_repair_after_edit(
            after_delete,
            changed_polygon_ids={polygon_id},
        )
        focus_id = resolve_focus_id_after_geometry_pass(before, polygon_id, after_delete)
        self._push_polygon_set_change(
            before,
            after_delete,
            "Delete vertex",
            select_polygon_id=focus_id,
        )
        return True

    def polygon_points(self, polygon_id: int) -> list[tuple[float, float]]:
        polygon = self._polygons.get(polygon_id)
        if polygon is None:
            return []
        return integer_points(polygon.points)

    def has_polygon(self, polygon_id: int | None) -> bool:
        return polygon_id is not None and polygon_id in self._polygons

    def begin_polygon_edit_preview(self, polygon_id: int) -> None:
        polygon = self._polygons.get(polygon_id)
        item = self._polygon_items.get(polygon_id)
        if polygon is None or item is None:
            self.cancel_polygon_edit_preview()
            return
        if self._polygon_edit_preview_id != polygon_id:
            self.cancel_polygon_edit_preview()
        self._polygon_edit_preview_id = polygon_id
        self._polygon_edit_preview_points = integer_points(polygon.points)
        self._polygon_edit_preview_origin_points = list(self._polygon_edit_preview_points)
        self._polygon_edit_preview_item.setPen(QPen(item.pen()))
        self._polygon_edit_preview_item.setBrush(QBrush(item.brush()))
        self._polygon_edit_preview_item.setPos(QPointF())
        self._polygon_edit_preview_item.setPath(QPainterPath(item.path()))
        item.hide()
        self._polygon_edit_preview_item.show()

    def _update_polygon_edit_preview_path(self) -> None:
        polygon_id = self._polygon_edit_preview_id
        points = self._polygon_edit_preview_points
        if polygon_id is None or points is None or polygon_id not in self._polygons:
            self._polygon_edit_preview_item.setPath(QPainterPath())
            return
        path = _closed_polygon_path(points)
        for cutout in self._cutout_polygons_for(polygon_id):
            path.addPath(_display_path_for_polygon(cutout, self._display_settings))
        path.setFillRule(Qt.FillRule.WindingFill)
        self._polygon_edit_preview_item.setPos(QPointF())
        self._polygon_edit_preview_item.setPath(path)

    def cancel_polygon_edit_preview(self) -> None:
        polygon_id = self._polygon_edit_preview_id
        self._polygon_edit_preview_id = None
        self._polygon_edit_preview_points = None
        self._polygon_edit_preview_origin_points = None
        self._polygon_edit_preview_item.hide()
        self._polygon_edit_preview_item.setPos(QPointF())
        self._polygon_edit_preview_item.setPath(QPainterPath())
        if polygon_id is None:
            return
        item = self._polygon_items.get(polygon_id)
        polygon = self._polygons.get(polygon_id)
        if item is None or polygon is None:
            return
        category = str(getattr(polygon, "category", "") or "")
        item.setVisible(
            self._polygon_category_visible.get(category, True)
            and self._polygon_overlays_visible
            and not _hole_display_hidden(polygon, self._polygons)
        )

    def polygon_edit_preview_points(self, polygon_id: int) -> list[tuple[float, float]] | None:
        if polygon_id != self._polygon_edit_preview_id or self._polygon_edit_preview_points is None:
            return None
        return list(self._polygon_edit_preview_points)

    def preview_vertex_move(
        self,
        polygon_id: int,
        vertex_index: int,
        point: QPointF,
    ) -> list[tuple[float, float]]:
        if self._polygon_edit_preview_id != polygon_id:
            self.begin_polygon_edit_preview(polygon_id)
        points = list(self._polygon_edit_preview_points or self.polygon_points(polygon_id))
        if vertex_index < 0 or vertex_index >= len(points):
            return points
        new_point = integer_point((point.x(), point.y()))
        closed_duplicate_endpoint = (
            len(points) > 2
            and hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) < 1e-5
        )
        points[vertex_index] = new_point
        if closed_duplicate_endpoint:
            if vertex_index == 0:
                points[-1] = new_point
            elif vertex_index == len(points) - 1:
                points[0] = new_point
        self._polygon_edit_preview_points = points
        self._update_polygon_edit_preview_path()
        return list(points)
