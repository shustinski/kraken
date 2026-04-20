from __future__ import annotations

from enum import Enum
from math import atan2, cos, hypot, pi, sin

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QShortcut,
    QUndoStack,
)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from .adapters.qt.image_conversion import cv_to_qimage
from .application.processing import DisplaySettings
from .commands import (
    AddPolygonCommand,
    AddVertexCommand,
    DeletePolygonCommand,
    DeleteVertexCommand,
    MovePolygonCommand,
    MoveVertexCommand,
)
from .domain import PolygonData, compute_polygon_metrics
from .graphics_items import EditablePolygonItem, VertexHandleItem
from .i18n import active_language, tr


class EditorTool(str, Enum):
    SELECT = "select"
    PAN = "pan"
    RULER = "ruler"
    ADD_POLYGON = "add_polygon"
    BRUSH = "brush"
    ADD_VIA = "add_via"
    ADD_VERTEX = "add_vertex"
    DELETE_VERTEX = "delete_vertex"
    MOVE_VERTEX = "move_vertex"
    DELETE_POLYGON = "delete_polygon"


class PolygonCreateMode(str, Enum):
    POINTS = "points"
    RECTANGLE = "rectangle"


class BrushMode(str, Enum):
    FREEFORM = "freeform"
    ANGLED = "angled"


class DeleteVertexMode(str, Enum):
    SINGLE = "single"
    AREA = "area"


class PolygonEditorScene(QGraphicsScene):
    polygonsChanged = pyqtSignal()
    activePolygonChanged = pyqtSignal(object)
    logRequested = pyqtSignal(str)

    def __init__(self, parent: QGraphicsView | None = None) -> None:
        super().__init__(parent)
        self.undo_stack = QUndoStack(self)
        self._ui_language = active_language()
        self._display_settings = DisplaySettings()
        self._polygons: dict[int, PolygonData] = {}
        self._polygon_items: dict[int, EditablePolygonItem] = {}
        self._selected_polygon_id: int | None = None
        self._next_polygon_id = 1
        self._polygon_overlays_visible = True

        self._image_item = QGraphicsPixmapItem()
        self._image_item.setZValue(0)
        self.addItem(self._image_item)

        self._pending_points: list[tuple[float, float]] = []
        self._pending_cursor: tuple[float, float] | None = None
        self._pending_path_item = QGraphicsPathItem()
        self._pending_path_item.setZValue(10)
        pending_pen = QPen(QColor("#F7B801"), 1.5, Qt.PenStyle.DashLine)
        pending_pen.setCosmetic(True)
        self._pending_path_item.setPen(pending_pen)
        self._pending_path_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.addItem(self._pending_path_item)
        self._preview_rect_item = QGraphicsPathItem()
        self._preview_rect_item.setZValue(11)
        preview_pen = QPen(QColor("#38BDF8"), 1.5, Qt.PenStyle.DashLine)
        preview_pen.setCosmetic(True)
        self._preview_rect_item.setPen(preview_pen)
        preview_brush = QColor("#38BDF8")
        preview_brush.setAlpha(48)
        self._preview_rect_item.setBrush(QBrush(preview_brush))
        self.addItem(self._preview_rect_item)
        self._via_cursor_item = QGraphicsPathItem()
        self._via_cursor_item.setZValue(12)
        via_cursor_pen = QPen(QColor("#A78BFA"), 1.5, Qt.PenStyle.DashLine)
        via_cursor_pen.setCosmetic(True)
        self._via_cursor_item.setPen(via_cursor_pen)
        via_cursor_brush = QColor("#A78BFA")
        via_cursor_brush.setAlpha(42)
        self._via_cursor_item.setBrush(QBrush(via_cursor_brush))
        self.addItem(self._via_cursor_item)
        self._via_cursor_item.hide()
        self._brush_cursor_item = QGraphicsEllipseItem()
        self._brush_cursor_item.setZValue(12)
        brush_cursor_pen = QPen(QColor("#4ADE80"), 1.5, Qt.PenStyle.DashLine)
        brush_cursor_pen.setCosmetic(True)
        self._brush_cursor_item.setPen(brush_cursor_pen)
        self._brush_cursor_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.addItem(self._brush_cursor_item)
        self._brush_cursor_item.hide()
        self._measurement_item = QGraphicsPathItem()
        self._measurement_item.setZValue(13)
        measurement_pen = QPen(QColor("#F59E0B"), 2.0, Qt.PenStyle.DashLine)
        measurement_pen.setCosmetic(True)
        self._measurement_item.setPen(measurement_pen)
        self._measurement_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.addItem(self._measurement_item)
        self._measurement_start_marker = QGraphicsEllipseItem()
        self._measurement_start_marker.setZValue(14)
        self._measurement_start_marker.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self._measurement_start_marker.setBrush(QBrush(QColor("#F59E0B")))
        marker_pen = QPen(QColor("#F8FAFC"), 1.0)
        marker_pen.setCosmetic(True)
        self._measurement_start_marker.setPen(marker_pen)
        self.addItem(self._measurement_start_marker)
        self._measurement_start_marker.hide()
        self._measurement_end_marker = QGraphicsEllipseItem()
        self._measurement_end_marker.setZValue(14)
        self._measurement_end_marker.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self._measurement_end_marker.setBrush(QBrush(QColor("#F59E0B")))
        self._measurement_end_marker.setPen(marker_pen)
        self.addItem(self._measurement_end_marker)
        self._measurement_end_marker.hide()
        self._measurement_label_item = QGraphicsSimpleTextItem()
        self._measurement_label_item.setZValue(15)
        self._measurement_label_item.setBrush(QBrush(QColor("#F8FAFC")))
        self._measurement_label_item.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.addItem(self._measurement_label_item)
        self._measurement_label_item.hide()
        self.setSceneRect(QRectF(0, 0, 1, 1))

    def set_pending_path_width(self, width: float, cosmetic: bool | None = None) -> None:
        pen = self._pending_path_item.pen()
        pen.setWidthF(max(1.0, float(width)))
        if cosmetic is not None:
            pen.setCosmetic(bool(cosmetic))
        self._pending_path_item.setPen(pen)

    def set_image(self, image) -> None:
        if image is None:
            self._image_item.setPixmap(QPixmap())
            self.setSceneRect(QRectF(0, 0, 1, 1))
            return
        pixmap = QPixmap.fromImage(cv_to_qimage(image))
        self._image_item.setPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))

    def set_ui_language(self, language: str | None) -> None:
        self._ui_language = active_language(language)

    def set_display_settings(self, settings: DisplaySettings) -> None:
        self._display_settings = settings
        self._refresh_all_items()

    def set_polygon_overlays_visible(self, visible: bool) -> None:
        self._polygon_overlays_visible = bool(visible)
        for item in self._polygon_items.values():
            item.setVisible(self._polygon_overlays_visible)
        self._pending_path_item.setVisible(self._polygon_overlays_visible)
        self._preview_rect_item.setVisible(self._polygon_overlays_visible)

    def polygon_overlays_visible(self) -> bool:
        return self._polygon_overlays_visible

    def get_polygons(self) -> list[PolygonData]:
        return [self._polygons[polygon_id].clone() for polygon_id in sorted(self._polygons)]

    def set_polygons(self, polygons: list[PolygonData]) -> None:
        self.undo_stack.clear()
        for item in list(self._polygon_items.values()):
            self.removeItem(item)
        self._polygon_items.clear()
        self._polygons.clear()
        self._selected_polygon_id = None
        self._next_polygon_id = 1
        for polygon in polygons:
            self._add_polygon_internal(polygon.clone(), emit_signal=False, refresh=False)
        if polygons:
            self._next_polygon_id = max(polygon.id for polygon in polygons) + 1
            self._selected_polygon_id = polygons[0].id
        self._refresh_all_items()
        self.polygonsChanged.emit()
        self.activePolygonChanged.emit(self._selected_polygon_id)

    def selected_polygon_id(self) -> int | None:
        return self._selected_polygon_id

    def select_polygon(self, polygon_id: int | None) -> None:
        if polygon_id is not None and polygon_id not in self._polygons:
            polygon_id = None
        self._selected_polygon_id = polygon_id
        self._refresh_all_items()
        self.activePolygonChanged.emit(polygon_id)

    def polygon_at(self, scene_pos: QPointF) -> int | None:
        for item in self.items(scene_pos):
            if isinstance(item, VertexHandleItem):
                return item.polygon_id
            if isinstance(item, EditablePolygonItem):
                return item.polygon_id
            parent = item.parentItem()
            if isinstance(parent, EditablePolygonItem):
                return parent.polygon_id
        return None

    def vertex_at(self, scene_pos: QPointF, tolerance: float) -> tuple[int, int] | None:
        candidate_ids = []
        if self._selected_polygon_id is not None:
            candidate_ids.append(self._selected_polygon_id)
        candidate_ids.extend([polygon_id for polygon_id in sorted(self._polygons) if polygon_id != self._selected_polygon_id])
        for polygon_id in candidate_ids:
            polygon = self._polygons[polygon_id]
            for index, (x_coord, y_coord) in enumerate(polygon.points):
                if hypot(scene_pos.x() - x_coord, scene_pos.y() - y_coord) <= tolerance:
                    return polygon_id, index
        return None

    def delete_polygon_at(self, scene_pos: QPointF) -> bool:
        polygon_id = self.polygon_at(scene_pos)
        if polygon_id is None:
            return False
        self.delete_polygon(polygon_id)
        return True

    def delete_polygon(self, polygon_id: int | None = None) -> bool:
        target_id = polygon_id if polygon_id is not None else self._selected_polygon_id
        if target_id is None or target_id not in self._polygons:
            return False
        self.undo_stack.push(DeletePolygonCommand(self, self._polygons[target_id]))
        return True

    def add_vertex_at(self, polygon_id: int, scene_pos: QPointF) -> bool:
        if polygon_id not in self._polygons:
            return False
        insert_index = self._nearest_segment_insert_index(polygon_id, scene_pos)
        self.undo_stack.push(AddVertexCommand(self, polygon_id, insert_index, (scene_pos.x(), scene_pos.y())))
        self.select_polygon(polygon_id)
        return True

    def delete_vertex_at(self, scene_pos: QPointF, tolerance: float) -> bool:
        hit = self.vertex_at(scene_pos, tolerance)
        if hit is None:
            return False
        polygon_id, vertex_index = hit
        polygon = self._polygons[polygon_id]
        if len(polygon.points) <= 3:
            self.logRequested.emit(tr("polygon_min_vertices_log", language=self._ui_language))
            return False
        self.undo_stack.push(DeleteVertexCommand(self, polygon_id, vertex_index, polygon.points[vertex_index]))
        self.select_polygon(polygon_id)
        return True

    def polygon_points(self, polygon_id: int) -> list[tuple[float, float]]:
        return [(float(x), float(y)) for x, y in self._polygons[polygon_id].points]

    def preview_vertex_move(self, polygon_id: int, vertex_index: int, point: QPointF) -> None:
        self._set_vertex_internal(polygon_id, vertex_index, (point.x(), point.y()), emit_signal=False)

    def preview_polygon_move(self, polygon_id: int, points: list[tuple[float, float]]) -> None:
        self._replace_polygon_points_internal(polygon_id, points, emit_signal=False)

    def start_pending_polygon(self) -> None:
        self._pending_points.clear()
        self._pending_cursor = None
        self._update_pending_path()

    def append_pending_point(self, scene_pos: QPointF) -> None:
        point = (scene_pos.x(), scene_pos.y())
        if self._pending_points and hypot(point[0] - self._pending_points[-1][0], point[1] - self._pending_points[-1][1]) < 1.0:
            return
        self._pending_points.append(point)
        self._update_pending_path()

    def update_pending_cursor(self, scene_pos: QPointF) -> None:
        if not self._pending_points:
            return
        self._pending_cursor = (scene_pos.x(), scene_pos.y())
        self._update_pending_path()

    def cancel_pending_polygon(self) -> None:
        self._pending_points.clear()
        self._pending_cursor = None
        self._update_pending_path()
        self.clear_preview_rect()

    def finish_pending_polygon(self) -> bool:
        if len(self._pending_points) < 3:
            self.cancel_pending_polygon()
            return False
        area, perimeter, bbox = compute_polygon_metrics(self._pending_points)
        polygon = PolygonData(
            id=self._next_polygon_id,
            points=[(float(x), float(y)) for x, y in self._pending_points],
            is_hole=False,
            parent_id=None,
            area=area,
            perimeter=perimeter,
            bbox=bbox,
        )
        self._add_or_merge_polygon(polygon)
        self.cancel_pending_polygon()
        return True

    def pending_last_point(self) -> QPointF | None:
        if not self._pending_points:
            return None
        x_coord, y_coord = self._pending_points[-1]
        return QPointF(x_coord, y_coord)

    def pending_points_snapshot(self) -> list[tuple[float, float]]:
        return [(float(x_coord), float(y_coord)) for x_coord, y_coord in self._pending_points]

    def has_pending_polygon(self) -> bool:
        return bool(self._pending_points)

    def set_preview_rect(self, start: QPointF, end: QPointF) -> None:
        rect = QRectF(start, end).normalized()
        path = QPainterPath()
        path.addRect(rect)
        self._preview_rect_item.setPath(path)

    def clear_preview_rect(self) -> None:
        self._preview_rect_item.setPath(QPainterPath())

    def set_measurement(self, start: QPointF, end: QPointF, label_text: str = "") -> None:
        path = QPainterPath()
        path.moveTo(start)
        path.lineTo(end)
        self._measurement_item.setPath(path)
        self._set_measurement_marker(self._measurement_start_marker, start)
        self._set_measurement_marker(self._measurement_end_marker, end)
        if label_text:
            self._measurement_label_item.setText(label_text)
            self._measurement_label_item.setPos(_measurement_label_position(start, end))
            self._measurement_label_item.show()
        else:
            self._measurement_label_item.hide()

    def clear_measurement(self) -> None:
        self._measurement_item.setPath(QPainterPath())
        self._measurement_start_marker.hide()
        self._measurement_end_marker.hide()
        self._measurement_label_item.hide()

    def set_brush_cursor(self, scene_pos: QPointF | None, thickness: float, visible: bool) -> None:
        if not visible or scene_pos is None:
            self._brush_cursor_item.hide()
            return
        radius = max(1.0, float(thickness)) / 2.0
        self._brush_cursor_item.setRect(QRectF(scene_pos.x() - radius, scene_pos.y() - radius, radius * 2.0, radius * 2.0))
        self._brush_cursor_item.show()

    def set_via_cursor(self, scene_pos: QPointF | None, width: float, height: float, visible: bool) -> None:
        if not visible or scene_pos is None:
            self._via_cursor_item.hide()
            return
        rect = _centered_rect(scene_pos, width, height)
        path = QPainterPath()
        path.addRect(rect)
        self._via_cursor_item.setPath(path)
        self._via_cursor_item.show()

    def hide_tool_cursors(self) -> None:
        self._brush_cursor_item.hide()
        self._via_cursor_item.hide()

    def add_via_at(self, scene_pos: QPointF, width: float, height: float) -> bool:
        rect = _centered_rect(scene_pos, width, height).normalized()
        if rect.width() < 1.0 or rect.height() < 1.0:
            return False
        points = [
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.right(), rect.bottom()),
            (rect.left(), rect.bottom()),
        ]
        area, perimeter, bbox = compute_polygon_metrics(points)
        polygon = PolygonData(
            id=self._next_polygon_id,
            points=points,
            is_hole=False,
            parent_id=None,
            category="via",
            shape_hint="box",
            area=area,
            perimeter=perimeter,
            bbox=bbox,
        )
        self.undo_stack.push(AddPolygonCommand(self, polygon))
        self.select_polygon(polygon.id)
        return True

    def add_rectangle_polygon(self, start: QPointF, end: QPointF, erase: bool = False) -> bool:
        rect = QRectF(start, end).normalized()
        if rect.width() < 1.0 or rect.height() < 1.0:
            self.clear_preview_rect()
            return False
        points = [
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.right(), rect.bottom()),
            (rect.left(), rect.bottom()),
        ]
        area, perimeter, bbox = compute_polygon_metrics(points)
        polygon = PolygonData(
            id=self._next_polygon_id,
            points=points,
            is_hole=False,
            parent_id=None,
            area=area,
            perimeter=perimeter,
            bbox=bbox,
        )
        if erase:
            self._subtract_shape_from_scene(points=polygon.points, thickness=None, label="Erase rectangle")
        else:
            self._add_or_merge_polygon(polygon, label="Add rectangle")
        self.clear_preview_rect()
        return True

    def add_brush_stroke(self, points: list[tuple[float, float]], thickness: float, erase: bool = False) -> bool:
        if len(points) < 2:
            self.cancel_pending_polygon()
            return False
        if erase:
            changed = self._subtract_shape_from_scene(points=points, thickness=thickness, label="Erase brush stroke")
            self.cancel_pending_polygon()
            return changed
        merged_polygons, overlapping_ids = self._merge_shape_into_scene(points=points, thickness=thickness)
        if not merged_polygons:
            self.cancel_pending_polygon()
            return False
        self.undo_stack.beginMacro("Add brush stroke")
        try:
            for polygon_id in overlapping_ids:
                self.undo_stack.push(DeletePolygonCommand(self, self._polygons[polygon_id]))
            for polygon in merged_polygons:
                self.undo_stack.push(AddPolygonCommand(self, polygon))
        finally:
            self.undo_stack.endMacro()
        self.select_polygon(merged_polygons[0].id)
        self.cancel_pending_polygon()
        return True

    def subtract_pending_polygon(self) -> bool:
        if len(self._pending_points) < 3:
            self.cancel_pending_polygon()
            return False
        changed = self._subtract_shape_from_scene(points=self._pending_points, thickness=None, label="Erase polygon")
        self.cancel_pending_polygon()
        return changed

    def delete_vertices_in_rect(self, rect: QRectF) -> int:
        normalized = rect.normalized()
        if normalized.width() < 1.0 and normalized.height() < 1.0:
            return 0
        candidate_ids = [self._selected_polygon_id] if self._selected_polygon_id is not None else sorted(self._polygons)
        deleted = 0
        self.undo_stack.beginMacro("Delete vertices in area")
        try:
            for polygon_id in candidate_ids:
                if polygon_id is None:
                    continue
                polygon = self._polygons.get(polygon_id)
                if polygon is None:
                    continue
                matching_indices = [
                    index
                    for index, (x_coord, y_coord) in enumerate(polygon.points)
                    if normalized.contains(QPointF(x_coord, y_coord))
                ]
                remaining = len(polygon.points)
                for vertex_index in reversed(matching_indices):
                    if remaining <= 3:
                        break
                    self.undo_stack.push(DeleteVertexCommand(self, polygon_id, vertex_index, polygon.points[vertex_index]))
                    remaining -= 1
                    deleted += 1
        finally:
            self.undo_stack.endMacro()
        return deleted

    def _add_or_merge_polygon(self, polygon: PolygonData, label: str = "Add polygon") -> None:
        merged_polygons, overlapping_ids = self._merge_shape_into_scene(points=polygon.points, thickness=None)
        if not merged_polygons:
            self.undo_stack.push(AddPolygonCommand(self, polygon))
            self.select_polygon(polygon.id)
            return
        self.undo_stack.beginMacro(label)
        try:
            for polygon_id in overlapping_ids:
                self.undo_stack.push(DeletePolygonCommand(self, self._polygons[polygon_id]))
            for merged_polygon in merged_polygons:
                self.undo_stack.push(AddPolygonCommand(self, merged_polygon))
        finally:
            self.undo_stack.endMacro()
        self.select_polygon(merged_polygons[0].id)

    def _subtract_shape_from_scene(
        self,
        *,
        points: list[tuple[float, float]],
        thickness: float | None,
        label: str,
    ) -> bool:
        if not points:
            return False
        shape_bbox = _bbox_from_points(points, padding=(int(round(thickness / 2.0)) + 2) if thickness else 2)
        overlapping_ids = self._find_overlapping_polygon_ids(points=points, thickness=thickness, shape_bbox=shape_bbox)
        if not overlapping_ids:
            return False
        region_boxes = [shape_bbox]
        render_ids = self._render_polygon_ids(overlapping_ids)
        touched_ids = self._touched_polygon_ids(render_ids, shape_bbox, points, thickness)
        preserved_polygons = self._preserved_polygons(render_ids, touched_ids, overlapping_ids)
        for polygon_id in render_ids:
            region_boxes.append(self._polygons[polygon_id].bbox)
        region_bbox = _union_bbox(region_boxes)
        remaining_polygons = self._apply_shape_mask(
            region_bbox=region_bbox,
            region_polygon_ids=render_ids,
            points=points,
            thickness=thickness,
            erase=True,
        )
        rebuilt_polygons = self._restore_preserved_polygons(remaining_polygons, render_ids, preserved_polygons)

        self.undo_stack.beginMacro(label)
        try:
            for polygon_id in render_ids:
                self.undo_stack.push(DeletePolygonCommand(self, self._polygons[polygon_id]))
            for polygon in rebuilt_polygons:
                self.undo_stack.push(AddPolygonCommand(self, polygon))
        finally:
            self.undo_stack.endMacro()

        if rebuilt_polygons:
            self.select_polygon(rebuilt_polygons[0].id)
        else:
            self.select_polygon(None)
        return True

    def _merge_shape_into_scene(
        self,
        *,
        points: list[tuple[float, float]],
        thickness: float | None,
    ) -> tuple[list[PolygonData], list[int]]:
        if not points:
            return [], []
        shape_bbox = _bbox_from_points(points, padding=(int(round(thickness / 2.0)) + 2) if thickness else 2)
        overlapping_ids = self._find_overlapping_polygon_ids(points=points, thickness=thickness, shape_bbox=shape_bbox)
        region_boxes = [shape_bbox]
        render_ids = self._render_polygon_ids(overlapping_ids)
        touched_ids = self._touched_polygon_ids(render_ids, shape_bbox, points, thickness)
        preserved_polygons = self._preserved_polygons(render_ids, touched_ids, overlapping_ids)
        for polygon_id in render_ids:
            region_boxes.append(self._polygons[polygon_id].bbox)
        region_bbox = _union_bbox(region_boxes)
        merged_contours = self._apply_shape_mask(
            region_bbox=region_bbox,
            region_polygon_ids=render_ids,
            points=points,
            thickness=thickness,
            erase=False,
        )
        if not merged_contours:
            return [], render_ids
        return self._restore_preserved_polygons(merged_contours, render_ids, preserved_polygons), render_ids

    def _find_overlapping_polygon_ids(
        self,
        *,
        points: list[tuple[float, float]],
        thickness: float | None,
        shape_bbox: tuple[int, int, int, int],
    ) -> list[int]:
        overlapping_ids: list[int] = []
        for polygon_id, polygon in self._polygons.items():
            if polygon.is_hole:
                continue
            if not _bboxes_intersect(shape_bbox, polygon.bbox):
                continue
            test_bbox = _union_bbox([shape_bbox, polygon.bbox])
            shape_mask, origin = self._render_shape_mask(test_bbox, points, thickness)
            polygon_mask = np.zeros_like(shape_mask)
            _render_polygon_collection_on_mask(
                polygon_mask,
                [self._polygons[item_id] for item_id in self._polygon_family_ids(polygon_id)],
                origin,
            )
            if np.any(cv2.bitwise_and(shape_mask, polygon_mask)):
                overlapping_ids.append(polygon_id)
        return overlapping_ids

    def _apply_shape_mask(
        self,
        region_bbox: tuple[int, int, int, int],
        region_polygon_ids: list[int],
        points: list[tuple[float, float]],
        thickness: float | None,
        *,
        erase: bool,
    ) -> list[PolygonData]:
        base_mask, origin = self._render_region_polygon_mask(region_bbox, region_polygon_ids)
        shape_mask, _ = self._render_shape_mask(region_bbox, points, thickness)
        if erase:
            result_mask = cv2.bitwise_and(base_mask, cv2.bitwise_not(shape_mask))
        else:
            result_mask = cv2.bitwise_or(base_mask, shape_mask)
        return _polygons_from_mask(result_mask, origin)

    def _render_region_polygon_mask(
        self,
        region_bbox: tuple[int, int, int, int],
        region_polygon_ids: list[int],
    ) -> tuple[np.ndarray, tuple[int, int]]:
        x_coord, y_coord, width, height = _clip_bbox_to_scene(region_bbox, self.sceneRect())
        mask = np.zeros((max(1, height), max(1, width)), dtype=np.uint8)
        origin = (x_coord, y_coord)
        _render_polygon_collection_on_mask(
            mask,
            [self._polygons[polygon_id] for polygon_id in region_polygon_ids],
            origin,
        )
        return mask, origin

    def _render_shape_mask(
        self,
        region_bbox: tuple[int, int, int, int],
        points: list[tuple[float, float]],
        thickness: float | None,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        x_coord, y_coord, width, height = _clip_bbox_to_scene(region_bbox, self.sceneRect())
        mask = np.zeros((max(1, height), max(1, width)), dtype=np.uint8)
        origin = (x_coord, y_coord)
        if thickness is None:
            _fill_polygon_on_mask(mask, points, origin)
        else:
            _draw_stroke_on_mask(mask, points, origin, thickness)
        return mask, origin

    def _allocate_polygon_ids(self, overlapping_ids: list[int], count: int) -> list[int]:
        ids: list[int] = []
        reusable_ids = sorted(overlapping_ids)
        for index in range(count):
            if index < len(reusable_ids):
                ids.append(reusable_ids[index])
            else:
                ids.append(self._next_polygon_id)
                self._next_polygon_id += 1
        return ids

    def _update_pending_path(self) -> None:
        path = QPainterPath()
        if self._pending_points:
            first = self._pending_points[0]
            path.moveTo(first[0], first[1])
            for point in self._pending_points[1:]:
                path.lineTo(point[0], point[1])
            if self._pending_cursor is not None:
                path.lineTo(self._pending_cursor[0], self._pending_cursor[1])
        self._pending_path_item.setPath(path)

    def _refresh_all_items(self) -> None:
        for polygon_id, item in self._polygon_items.items():
            item.update_from_polygon(
                self._polygons[polygon_id],
                self._display_settings,
                selected=polygon_id == self._selected_polygon_id,
                cutout_polygons=self._cutout_polygons_for(polygon_id),
            )

    def _cutout_polygons_for(self, polygon_id: int) -> list[PolygonData]:
        polygon = self._polygons.get(polygon_id)
        if polygon is None or polygon.is_hole:
            return []
        return [
            child.clone()
            for child in self._polygons.values()
            if child.parent_id == polygon_id and child.is_hole
        ]

    def _render_polygon_ids(self, overlapping_ids: list[int]) -> list[int]:
        render_ids: list[int] = []
        for polygon_id in overlapping_ids:
            for family_id in self._polygon_family_ids(polygon_id):
                if family_id not in render_ids:
                    render_ids.append(family_id)
        return sorted(render_ids)

    def _touched_polygon_ids(
        self,
        candidate_ids: list[int],
        shape_bbox: tuple[int, int, int, int],
        points: list[tuple[float, float]],
        thickness: float | None,
    ) -> set[int]:
        touched_ids: set[int] = set()
        for polygon_id in candidate_ids:
            polygon = self._polygons.get(polygon_id)
            if polygon is None or not _bboxes_intersect(shape_bbox, polygon.bbox):
                continue
            test_bbox = _union_bbox([shape_bbox, polygon.bbox])
            shape_mask, origin = self._render_shape_mask(test_bbox, points, thickness)
            polygon_mask = np.zeros_like(shape_mask)
            _fill_polygon_on_mask(polygon_mask, polygon.points, origin)
            if np.any(cv2.bitwise_and(shape_mask, polygon_mask)):
                touched_ids.add(polygon_id)
        return touched_ids

    def _preserved_polygons(
        self,
        render_ids: list[int],
        touched_ids: set[int],
        root_ids: list[int],
    ) -> list[PolygonData]:
        root_id_set = set(root_ids)
        return [
            self._polygons[polygon_id].clone()
            for polygon_id in render_ids
            if polygon_id not in root_id_set and polygon_id not in touched_ids
        ]

    def _polygon_family_ids(self, polygon_id: int) -> list[int]:
        family_ids: list[int] = []
        pending = [polygon_id]
        while pending:
            current_id = pending.pop()
            if current_id in family_ids or current_id not in self._polygons:
                continue
            family_ids.append(current_id)
            pending.extend(
                child_id
                for child_id, polygon in self._polygons.items()
                if polygon.parent_id == current_id
            )
        return sorted(family_ids)

    def _assign_polygon_ids(
        self,
        polygons: list[PolygonData],
        replaced_ids: list[int],
        *,
        reserved_ids: set[int] | None = None,
    ) -> list[PolygonData]:
        reserved = reserved_ids or set()
        reusable_ids = [polygon_id for polygon_id in sorted(replaced_ids) if polygon_id not in reserved]
        allocated_ids = self._allocate_polygon_ids(reusable_ids, len(polygons))
        id_map = {
            polygon.id: allocated_id
            for polygon, allocated_id in zip(polygons, allocated_ids, strict=False)
        }
        for polygon, allocated_id in zip(polygons, allocated_ids, strict=False):
            polygon.parent_id = None if polygon.parent_id is None else id_map.get(polygon.parent_id)
            polygon.id = allocated_id
        return polygons

    def _restore_preserved_polygons(
        self,
        rebuilt_polygons: list[PolygonData],
        deleted_ids: list[int],
        preserved_polygons: list[PolygonData],
    ) -> list[PolygonData]:
        if not preserved_polygons:
            return self._assign_polygon_ids(rebuilt_polygons, deleted_ids)
        filtered_rebuilt = [
            polygon
            for polygon in rebuilt_polygons
            if not self._matches_any_preserved_polygon(polygon, preserved_polygons)
        ]
        assigned_rebuilt = self._assign_polygon_ids(
            filtered_rebuilt,
            deleted_ids,
            reserved_ids={polygon.id for polygon in preserved_polygons},
        )
        restored_polygons = assigned_rebuilt + [polygon.clone() for polygon in preserved_polygons]
        self._repair_preserved_parent_links(restored_polygons, preserved_polygons)
        return sorted(restored_polygons, key=lambda polygon: polygon.id)

    def _matches_any_preserved_polygon(self, polygon: PolygonData, preserved_polygons: list[PolygonData]) -> bool:
        return any(
            polygon.is_hole == preserved.is_hole and _bboxes_intersect(polygon.bbox, preserved.bbox)
            for preserved in preserved_polygons
        )

    def _repair_preserved_parent_links(
        self,
        polygons: list[PolygonData],
        preserved_polygons: list[PolygonData],
    ) -> None:
        non_hole_polygons = [polygon for polygon in polygons if not polygon.is_hole]
        for preserved in preserved_polygons:
            restored = next((polygon for polygon in polygons if polygon.id == preserved.id), None)
            if restored is None:
                continue
            parent = _smallest_containing_polygon(restored, non_hole_polygons)
            restored.parent_id = None if parent is None else parent.id

    def _create_polygon_snapshot(self, polygon_id: int, points: list[tuple[float, float]]) -> PolygonData:
        existing = self._polygons[polygon_id]
        area, perimeter, bbox = compute_polygon_metrics(points)
        return PolygonData(
            id=existing.id,
            points=[(float(x), float(y)) for x, y in points],
            is_hole=existing.is_hole,
            parent_id=existing.parent_id,
            category=existing.category,
            shape_hint=existing.shape_hint,
            area=area,
            perimeter=perimeter,
            bbox=bbox,
        )

    def _add_polygon_internal(self, polygon: PolygonData, emit_signal: bool = True, refresh: bool = True) -> None:
        if polygon.id in self._polygon_items:
            self._remove_polygon_internal(polygon.id, emit_signal=False, refresh=False)
        self._polygons[polygon.id] = polygon.clone()
        self._next_polygon_id = max(self._next_polygon_id, polygon.id + 1)
        item = EditablePolygonItem(self._polygons[polygon.id], self._display_settings)
        item.setVisible(self._polygon_overlays_visible)
        self._polygon_items[polygon.id] = item
        self.addItem(item)
        if refresh:
            self._refresh_all_items()
        if emit_signal:
            self.polygonsChanged.emit()

    def _remove_polygon_internal(self, polygon_id: int, emit_signal: bool = True, refresh: bool = True) -> None:
        item = self._polygon_items.pop(polygon_id, None)
        self._polygons.pop(polygon_id, None)
        if item is not None:
            self.removeItem(item)
        if self._selected_polygon_id == polygon_id:
            self._selected_polygon_id = None
            self.activePolygonChanged.emit(None)
        if refresh:
            self._refresh_all_items()
        if emit_signal:
            self.polygonsChanged.emit()

    def _replace_polygon_points_internal(self, polygon_id: int, points: list[tuple[float, float]], emit_signal: bool = True) -> None:
        if polygon_id not in self._polygons:
            return
        self._polygons[polygon_id] = self._create_polygon_snapshot(polygon_id, points)
        self._refresh_all_items()
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
        points[vertex_index] = (float(point[0]), float(point[1]))
        self._replace_polygon_points_internal(polygon_id, points, emit_signal=emit_signal)

    def _insert_vertex_internal(
        self,
        polygon_id: int,
        insert_index: int,
        point: tuple[float, float],
        emit_signal: bool = True,
    ) -> None:
        points = self.polygon_points(polygon_id)
        insert_at = max(0, min(len(points), insert_index))
        points.insert(insert_at, (float(point[0]), float(point[1])))
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


class PolygonEditorView(QGraphicsView):
    polygonsEdited = pyqtSignal()
    activePolygonChanged = pyqtSignal(object)
    logRequested = pyqtSignal(str)
    imageClicked = pyqtSignal(float, float)
    rulerMeasurementChanged = pyqtSignal(str)
    toolChanged = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        self._editor_scene = PolygonEditorScene()
        super().__init__(self._editor_scene, parent)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QBrush(QColor("#171B22")))
        self.setMouseTracking(True)

        self._tool = EditorTool.SELECT
        self._polygon_create_mode = PolygonCreateMode.POINTS
        self._brush_mode = BrushMode.FREEFORM
        self._brush_thickness = 12.0
        self._via_width = 12.0
        self._via_height = 12.0
        self._delete_vertex_mode = DeleteVertexMode.SINGLE
        self._drag_kind: str | None = None
        self._drag_polygon_id: int | None = None
        self._drag_vertex_index: int | None = None
        self._drag_origin_points: list[tuple[float, float]] | None = None
        self._drag_start_scene_pos: QPointF | None = None
        self._last_pointer_scene_pos: QPointF | None = None
        self._drag_erases = False
        self._pending_polygon_erases: bool | None = None
        self._middle_button_hides_overlays = False
        self._image_click_mode = False

        self._editor_scene.polygonsChanged.connect(self.polygonsEdited.emit)
        self._editor_scene.activePolygonChanged.connect(self.activePolygonChanged.emit)
        self._editor_scene.logRequested.connect(self.logRequested.emit)

        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self.redo)

    @property
    def undo_stack(self) -> QUndoStack:
        return self._editor_scene.undo_stack

    @property
    def current_tool(self) -> EditorTool:
        return self._tool

    def set_tool(self, tool: EditorTool) -> None:
        self._tool = tool
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if tool == EditorTool.PAN else QGraphicsView.DragMode.NoDrag)
        if tool == EditorTool.ADD_POLYGON:
            self._editor_scene.set_pending_path_width(1.5, cosmetic=True)
        elif tool == EditorTool.BRUSH:
            self._editor_scene.set_pending_path_width(self._brush_thickness, cosmetic=False)
        if tool not in (EditorTool.ADD_POLYGON, EditorTool.BRUSH):
            self._editor_scene.cancel_pending_polygon()
            self._pending_polygon_erases = None
        if tool != EditorTool.DELETE_VERTEX:
            self._editor_scene.clear_preview_rect()
        if tool != EditorTool.RULER:
            self._editor_scene.clear_measurement()
            self.rulerMeasurementChanged.emit("")
        self._update_tool_cursors()
        self.toolChanged.emit(tool)

    def set_polygon_create_mode(self, mode: PolygonCreateMode) -> None:
        self._polygon_create_mode = mode
        self._editor_scene.cancel_pending_polygon()
        self._pending_polygon_erases = None

    def set_brush_mode(self, mode: BrushMode) -> None:
        self._brush_mode = mode
        self._editor_scene.cancel_pending_polygon()

    def set_brush_thickness(self, thickness: float) -> None:
        self._brush_thickness = max(1.0, float(thickness))
        if self._tool == EditorTool.BRUSH:
            self._editor_scene.set_pending_path_width(self._brush_thickness, cosmetic=False)
        self._update_tool_cursors()

    def set_via_size(self, width: float, height: float) -> None:
        self._via_width = max(1.0, float(width))
        self._via_height = max(1.0, float(height))
        self._update_tool_cursors()

    def set_delete_vertex_mode(self, mode: DeleteVertexMode) -> None:
        self._delete_vertex_mode = mode
        self._editor_scene.clear_preview_rect()

    def set_image(self, image) -> None:
        previous_rect = QRectF(self._editor_scene.sceneRect())
        self._editor_scene.set_image(image)
        previous_was_placeholder = previous_rect.width() <= 1.0 and previous_rect.height() <= 1.0
        if previous_was_placeholder:
            self.fit_to_view()

    def set_polygons(self, polygons: list[PolygonData]) -> None:
        self._editor_scene.set_polygons(polygons)

    def get_polygons(self) -> list[PolygonData]:
        return self._editor_scene.get_polygons()

    def set_display_settings(self, settings: DisplaySettings) -> None:
        self._editor_scene.set_display_settings(settings)

    def set_ui_language(self, language: str | None) -> None:
        self._editor_scene.set_ui_language(language)

    def set_image_click_mode(self, enabled: bool) -> None:
        self._image_click_mode = bool(enabled)

    def fit_to_view(self) -> None:
        rect = self._editor_scene.sceneRect()
        if rect.width() > 0 and rect.height() > 0:
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_in(self) -> None:
        self.scale(1.15, 1.15)

    def zoom_out(self) -> None:
        self.scale(1 / 1.15, 1 / 1.15)

    def undo(self) -> None:
        self.undo_stack.undo()

    def redo(self) -> None:
        self.undo_stack.redo()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta()
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            if delta.y() == 0:
                event.accept()
                return
            factor = 1.15 ** (delta.y() / 120.0)
            self.scale(factor, factor)
            self._update_tool_cursors()
            event.accept()
            return
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            delta_value = delta.x() if delta.x() else delta.y()
            scrollbar = self.horizontalScrollBar()
            scrollbar.setValue(scrollbar.value() - delta_value)
            event.accept()
            return
        super().wheelEvent(event)
        self._update_tool_cursors()
        event.accept()

    def mousePressEvent(self, event) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        self._last_pointer_scene_pos = scene_pos
        tolerance = self._scene_tolerance(8)

        if self._image_click_mode and event.button() == Qt.MouseButton.LeftButton:
            self.imageClicked.emit(scene_pos.x(), scene_pos.y())
            event.accept()
            return

        if self._tool == EditorTool.PAN:
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_button_hides_overlays = True
            self._editor_scene.set_polygon_overlays_visible(False)
            event.accept()
            return

        if self._tool == EditorTool.ADD_POLYGON:
            if self._polygon_create_mode == PolygonCreateMode.RECTANGLE and event.button() == Qt.MouseButton.LeftButton:
                self._drag_kind = "rect_polygon"
                self._drag_start_scene_pos = scene_pos
                self._drag_erases = False
                self._editor_scene.set_preview_rect(scene_pos, scene_pos)
                event.accept()
                return
            if self._polygon_create_mode == PolygonCreateMode.RECTANGLE and event.button() == Qt.MouseButton.RightButton:
                self._drag_kind = "rect_polygon"
                self._drag_start_scene_pos = scene_pos
                self._drag_erases = True
                self._editor_scene.set_preview_rect(scene_pos, scene_pos)
                event.accept()
                return
            if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                requested_erase = event.button() == Qt.MouseButton.RightButton
                if self._editor_scene.has_pending_polygon():
                    if self._pending_polygon_erases is None:
                        self._pending_polygon_erases = requested_erase
                    elif requested_erase != self._pending_polygon_erases:
                        self._finish_pending_polygon()
                        event.accept()
                        return
                else:
                    self._pending_polygon_erases = requested_erase
                self._editor_scene.append_pending_point(scene_pos)
                event.accept()
                return

        if self._tool == EditorTool.BRUSH and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._drag_kind = "brush"
            self._drag_start_scene_pos = scene_pos
            self._drag_erases = event.button() == Qt.MouseButton.RightButton
            self._editor_scene.start_pending_polygon()
            self._editor_scene.set_pending_path_width(self._brush_thickness, cosmetic=False)
            self._append_brush_point(scene_pos)
            event.accept()
            return

        if self._tool == EditorTool.ADD_VIA and event.button() == Qt.MouseButton.LeftButton:
            self._editor_scene.add_via_at(scene_pos, self._via_width, self._via_height)
            self._update_tool_cursors()
            event.accept()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        if self._tool == EditorTool.RULER:
            self._drag_kind = "ruler"
            self._drag_start_scene_pos = scene_pos
            measurement_text = self._format_ruler_measurement(scene_pos, scene_pos)
            self._editor_scene.set_measurement(scene_pos, scene_pos, measurement_text)
            self.rulerMeasurementChanged.emit(measurement_text)
            event.accept()
            return

        if self._tool == EditorTool.DELETE_POLYGON:
            self._editor_scene.delete_polygon_at(scene_pos)
            event.accept()
            return

        if self._tool == EditorTool.ADD_VERTEX:
            polygon_id = self._editor_scene.selected_polygon_id() or self._editor_scene.polygon_at(scene_pos)
            if polygon_id is not None:
                self._editor_scene.add_vertex_at(polygon_id, scene_pos)
            event.accept()
            return

        if self._tool == EditorTool.DELETE_VERTEX:
            if self._delete_vertex_mode == DeleteVertexMode.AREA:
                self._drag_kind = "delete_area"
                self._drag_start_scene_pos = scene_pos
                self._editor_scene.set_preview_rect(scene_pos, scene_pos)
                event.accept()
                return
            self._editor_scene.delete_vertex_at(scene_pos, tolerance)
            event.accept()
            return

        if self._tool == EditorTool.MOVE_VERTEX:
            hit = self._editor_scene.vertex_at(scene_pos, tolerance)
            if hit is not None:
                polygon_id, vertex_index = hit
                self._editor_scene.select_polygon(polygon_id)
                self._drag_kind = "vertex"
                self._drag_polygon_id = polygon_id
                self._drag_vertex_index = vertex_index
                self._drag_origin_points = self._editor_scene.polygon_points(polygon_id)
                self._drag_start_scene_pos = scene_pos
            event.accept()
            return

        polygon_id = self._editor_scene.polygon_at(scene_pos)
        self._editor_scene.select_polygon(polygon_id)
        if polygon_id is not None:
            self._drag_kind = "polygon"
            self._drag_polygon_id = polygon_id
            self._drag_origin_points = self._editor_scene.polygon_points(polygon_id)
            self._drag_start_scene_pos = scene_pos
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        self._last_pointer_scene_pos = scene_pos
        self._update_tool_cursors()
        if self._tool == EditorTool.PAN:
            super().mouseMoveEvent(event)
            return
        if self._tool == EditorTool.ADD_POLYGON and self._polygon_create_mode == PolygonCreateMode.POINTS:
            self._editor_scene.update_pending_cursor(scene_pos)
            event.accept()
            return
        if self._drag_kind == "rect_polygon" and self._drag_start_scene_pos is not None:
            self._editor_scene.set_preview_rect(self._drag_start_scene_pos, scene_pos)
            event.accept()
            return
        if self._drag_kind == "ruler" and self._drag_start_scene_pos is not None:
            target_pos = self._ruler_target(self._drag_start_scene_pos, scene_pos, event.modifiers())
            measurement_text = self._format_ruler_measurement(self._drag_start_scene_pos, target_pos)
            self._editor_scene.set_measurement(self._drag_start_scene_pos, target_pos, measurement_text)
            self.rulerMeasurementChanged.emit(measurement_text)
            event.accept()
            return
        if self._drag_kind == "delete_area" and self._drag_start_scene_pos is not None:
            self._editor_scene.set_preview_rect(self._drag_start_scene_pos, scene_pos)
            event.accept()
            return
        if self._drag_kind == "brush":
            if self._brush_mode == BrushMode.ANGLED and self._drag_start_scene_pos is not None:
                self._editor_scene.update_pending_cursor(_snap_to_45(self._drag_start_scene_pos, scene_pos))
            else:
                self._append_brush_point(scene_pos)
            event.accept()
            return
        if self._drag_kind == "vertex" and self._drag_polygon_id is not None and self._drag_vertex_index is not None:
            self._editor_scene.preview_vertex_move(self._drag_polygon_id, self._drag_vertex_index, scene_pos)
            event.accept()
            return
        if self._drag_kind == "polygon" and self._drag_polygon_id is not None and self._drag_origin_points is not None and self._drag_start_scene_pos is not None:
            dx = scene_pos.x() - self._drag_start_scene_pos.x()
            dy = scene_pos.y() - self._drag_start_scene_pos.y()
            moved = [(x_coord + dx, y_coord + dy) for x_coord, y_coord in self._drag_origin_points]
            self._editor_scene.preview_polygon_move(self._drag_polygon_id, moved)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._tool == EditorTool.PAN:
            super().mouseReleaseEvent(event)
            return
        if event.button() == Qt.MouseButton.MiddleButton and self._middle_button_hides_overlays:
            self._middle_button_hides_overlays = False
            self._editor_scene.set_polygon_overlays_visible(True)
            event.accept()
            return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton) and self._drag_kind is not None:
            if self._drag_kind == "brush":
                release_pos = self.mapToScene(event.position().toPoint())
                if self._brush_mode == BrushMode.ANGLED and self._drag_start_scene_pos is not None:
                    end_point = _snap_to_45(self._drag_start_scene_pos, release_pos)
                    brush_points = [
                        (self._drag_start_scene_pos.x(), self._drag_start_scene_pos.y()),
                        (end_point.x(), end_point.y()),
                    ]
                else:
                    self._append_brush_point(release_pos)
                    brush_points = self._editor_scene.pending_points_snapshot()
                self._editor_scene.add_brush_stroke(brush_points, self._brush_thickness, erase=self._drag_erases)
            elif self._drag_kind == "rect_polygon" and self._drag_start_scene_pos is not None:
                self._editor_scene.add_rectangle_polygon(
                    self._drag_start_scene_pos,
                    self.mapToScene(event.position().toPoint()),
                    erase=self._drag_erases,
                )
            elif self._drag_kind == "ruler" and self._drag_start_scene_pos is not None:
                release_pos = self.mapToScene(event.position().toPoint())
                target_pos = self._ruler_target(self._drag_start_scene_pos, release_pos, event.modifiers())
                measurement_text = self._format_ruler_measurement(self._drag_start_scene_pos, target_pos)
                self._editor_scene.set_measurement(self._drag_start_scene_pos, target_pos, measurement_text)
                self.rulerMeasurementChanged.emit(measurement_text)
            elif self._drag_kind == "delete_area" and self._drag_start_scene_pos is not None:
                self._editor_scene.delete_vertices_in_rect(QRectF(self._drag_start_scene_pos, self.mapToScene(event.position().toPoint())))
                self._editor_scene.clear_preview_rect()
            elif self._drag_kind == "vertex" and self._drag_polygon_id is not None and self._drag_vertex_index is not None and self._drag_origin_points is not None:
                new_points = self._editor_scene.polygon_points(self._drag_polygon_id)
                old_point = self._drag_origin_points[self._drag_vertex_index]
                new_point = new_points[self._drag_vertex_index]
                if _points_different(old_point, new_point):
                    self.undo_stack.push(
                        MoveVertexCommand(
                            self._editor_scene,
                            self._drag_polygon_id,
                            self._drag_vertex_index,
                            old_point,
                            new_point,
                        )
                    )
            elif self._drag_kind == "polygon" and self._drag_polygon_id is not None and self._drag_origin_points is not None:
                new_points = self._editor_scene.polygon_points(self._drag_polygon_id)
                if _polygon_points_different(self._drag_origin_points, new_points):
                    self.undo_stack.push(
                        MovePolygonCommand(
                            self._editor_scene,
                            self._drag_polygon_id,
                            self._drag_origin_points,
                            new_points,
                        )
                    )
            self._drag_kind = None
            self._drag_polygon_id = None
            self._drag_vertex_index = None
            self._drag_origin_points = None
            self._drag_start_scene_pos = None
            self._drag_erases = False
            self._update_tool_cursors()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if (
            self._tool == EditorTool.ADD_POLYGON
            and self._polygon_create_mode == PolygonCreateMode.POINTS
            and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)
        ):
            self._finish_pending_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        if (
            event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return)
            and self._tool == EditorTool.ADD_POLYGON
            and self._polygon_create_mode == PolygonCreateMode.POINTS
        ):
            self._finish_pending_polygon()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self._editor_scene.cancel_pending_polygon()
            self._editor_scene.clear_measurement()
            if self._tool == EditorTool.RULER:
                self.rulerMeasurementChanged.emit("")
            self._drag_kind = None
            self._drag_erases = False
            self._pending_polygon_erases = None
            self._update_tool_cursors()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Delete:
            self._editor_scene.delete_polygon()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Shift and self._drag_kind == "ruler" and self._drag_start_scene_pos is not None and self._last_pointer_scene_pos is not None:
            target_pos = self._ruler_target(self._drag_start_scene_pos, self._last_pointer_scene_pos, event.modifiers())
            measurement_text = self._format_ruler_measurement(self._drag_start_scene_pos, target_pos)
            self._editor_scene.set_measurement(self._drag_start_scene_pos, target_pos, measurement_text)
            self.rulerMeasurementChanged.emit(measurement_text)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Shift and self._drag_kind == "ruler" and self._drag_start_scene_pos is not None and self._last_pointer_scene_pos is not None:
            target_pos = self._ruler_target(self._drag_start_scene_pos, self._last_pointer_scene_pos, event.modifiers())
            measurement_text = self._format_ruler_measurement(self._drag_start_scene_pos, target_pos)
            self._editor_scene.set_measurement(self._drag_start_scene_pos, target_pos, measurement_text)
            self.rulerMeasurementChanged.emit(measurement_text)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._editor_scene.hide_tool_cursors()
        super().leaveEvent(event)

    def _scene_tolerance(self, pixels: int) -> float:
        start = self.mapToScene(QPoint(0, 0))
        end = self.mapToScene(QPoint(pixels, 0))
        return max(1.0, abs(end.x() - start.x()))

    def _append_brush_point(self, scene_pos: QPointF) -> None:
        target = scene_pos
        if self._brush_mode == BrushMode.ANGLED:
            last_point = self._editor_scene.pending_last_point()
            if last_point is not None:
                target = _snap_to_45(last_point, scene_pos)
        self._editor_scene.append_pending_point(target)

    def _update_tool_cursors(self) -> None:
        self._editor_scene.set_brush_cursor(
            self._last_pointer_scene_pos,
            self._brush_thickness,
            self._tool == EditorTool.BRUSH,
        )
        self._editor_scene.set_via_cursor(
            self._last_pointer_scene_pos,
            self._via_width,
            self._via_height,
            self._tool == EditorTool.ADD_VIA,
        )

    def _finish_pending_polygon(self) -> None:
        if self._pending_polygon_erases:
            self._editor_scene.subtract_pending_polygon()
        else:
            self._editor_scene.finish_pending_polygon()
        self._pending_polygon_erases = None

    def _ruler_target(self, start: QPointF, target: QPointF, modifiers: Qt.KeyboardModifier | Qt.KeyboardModifiers) -> QPointF:
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            return _snap_to_45(start, target)
        return QPointF(target)

    def _format_ruler_measurement(self, start: QPointF, end: QPointF) -> str:
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        distance = hypot(dx, dy)
        return (
            f"L={distance:.1f}px, dX={dx:.1f}, dY={dy:.1f}"
            if self._editor_scene._ui_language != "ru"
            else f"L={distance:.1f}px, dX={dx:.1f}, dY={dy:.1f}"
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


def _fill_polygon_on_mask(mask: np.ndarray, points: list[tuple[float, float]], origin: tuple[int, int], value: int = 255) -> None:
    shifted = np.asarray(
        [[int(round(x_coord - origin[0])), int(round(y_coord - origin[1]))] for x_coord, y_coord in points],
        dtype=np.int32,
    )
    if shifted.shape[0] >= 3:
        cv2.fillPoly(mask, [shifted.reshape((-1, 1, 2))], int(value))


def _draw_polygon_outline_on_mask(mask: np.ndarray, points: list[tuple[float, float]], origin: tuple[int, int], value: int = 255) -> None:
    shifted = np.asarray(
        [[int(round(x_coord - origin[0])), int(round(y_coord - origin[1]))] for x_coord, y_coord in points],
        dtype=np.int32,
    )
    if shifted.shape[0] >= 3:
        cv2.polylines(mask, [shifted.reshape((-1, 1, 2))], True, int(value), thickness=1, lineType=cv2.LINE_8)


def _draw_stroke_on_mask(mask: np.ndarray, points: list[tuple[float, float]], origin: tuple[int, int], thickness: float) -> None:
    shifted = [
        (int(round(x_coord - origin[0])), int(round(y_coord - origin[1])))
        for x_coord, y_coord in points
    ]
    line_width = max(1, int(round(thickness)))
    radius = max(1, line_width // 2)
    for start, end in zip(shifted, shifted[1:], strict=False):
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
        points = [
            (float(point[0][0] + origin[0]), float(point[0][1] + origin[1]))
            for point in approx
        ]
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
        for polygon_id, (contour_index, _parent_index, _depth, _points, _area, _perimeter, _bbox)
        in enumerate(intermediates, start=1)
    }
    polygons: list[PolygonData] = []
    for polygon_id, (contour_index, parent_index, depth, points, area, perimeter, bbox) in enumerate(intermediates, start=1):
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
