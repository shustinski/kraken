from __future__ import annotations

from math import hypot

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QPainterPath,
    QPen,
    QPixmap,
    QTransform,
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

from ..adapters.qt.image_conversion import cv_to_qimage
from ..application.processing import DisplaySettings
from ..commands import (
    AddPolygonCommand,
    AddVertexCommand,
    DeletePolygonCommand,
    DeleteVertexCommand,
)
from ..domain import PolygonData, compute_polygon_metrics
from ..graphics_items import EditablePolygonItem, VertexHandleItem, _display_path_for_polygon
from ..i18n import active_language, tr
from .geometry import (
    _bbox_from_points,
    _bboxes_intersect,
    _centered_rect,
    _clip_bbox_to_scene,
    _distance_to_segment,
    _draw_stroke_on_mask,
    _fill_polygon_on_mask,
    is_valid_closed_polygon_ring,
    is_valid_open_polyline_last_edge,
    _measurement_label_position,
    _polygon_data_rect,
    _polygons_from_mask,
    _render_polygon_collection_on_mask,
    resolve_conductor_hover_target_id,
    _smallest_containing_polygon,
    _stable_object_color,
    _union_bbox,
)


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
        self._selected_polygon_ids: set[int] = set()
        self._next_polygon_id = 1
        self._polygon_overlays_visible = True
        self._polygon_category_visible: dict[str, bool] = {}

        self._image_item = QGraphicsPixmapItem()
        self._image_item.setZValue(0)
        self.addItem(self._image_item)
        self._image_rect = QRectF(0, 0, 1, 1)
        self._neighbor_frame_items: list[QGraphicsPixmapItem] = []
        self._neighbor_frame_paths: dict[QGraphicsPixmapItem, str] = {}
        self._neighbor_grid_bounds: QRectF | None = None
        self._debug_candidate_items: list[QGraphicsPathItem | QGraphicsSimpleTextItem] = []
        self._metal_overlay_items: list[QGraphicsPathItem] = []
        self._extra_layer_items: list[QGraphicsPixmapItem] = []
        self._gradient_overlay_item = QGraphicsPixmapItem()
        self._gradient_overlay_item.setZValue(0.9)
        self._gradient_overlay_item.setOpacity(0.45)
        self.addItem(self._gradient_overlay_item)
        self._gradient_overlay_item.hide()
        self._random_object_colors_enabled = False
        self._object_colors: dict[int, str] = {}
        self._hover_conductor_polygon_id: int | None = None

        self._main_frame_item = QGraphicsPathItem()
        self._main_frame_item.setZValue(2)
        main_frame_pen = QPen(QColor("#FACC15"), 2.0)
        main_frame_pen.setCosmetic(True)
        self._main_frame_item.setPen(main_frame_pen)
        self._main_frame_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.addItem(self._main_frame_item)
        self._main_frame_item.hide()

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
            self._image_rect = QRectF(0, 0, 1, 1)
            self._main_frame_item.setPath(QPainterPath())
            self._main_frame_item.hide()
            self.clear_neighbor_frames()
            self.set_debug_candidates([])
            self.set_metal_overlays({}, {})
            self._update_scene_rect()
            return
        pixmap = QPixmap.fromImage(cv_to_qimage(image))
        self._image_item.setPixmap(pixmap)
        self._image_rect = QRectF(pixmap.rect())
        self._update_main_frame()
        self._update_scene_rect()

    def main_image_rect(self) -> QRectF:
        return QRectF(self._image_rect)

    def clear_neighbor_frames(self) -> None:
        for item in self._neighbor_frame_items:
            self.removeItem(item)
        self._neighbor_frame_items.clear()
        self._neighbor_frame_paths.clear()
        self._neighbor_grid_bounds = None
        self._main_frame_item.hide()
        self._update_scene_rect()

    def set_neighbor_frames(
        self,
        frames: list[tuple[int, int, object, str]],
        opacity: float,
        overlap_pixels: int = 0,
        show_main_frame: bool = True,
    ) -> None:
        self.clear_neighbor_frames()
        self._main_frame_item.setVisible(bool(show_main_frame))
        if self._image_rect.width() <= 1.0 or self._image_rect.height() <= 1.0:
            return
        main_width = float(self._image_rect.width())
        main_height = float(self._image_rect.height())
        overlap = max(0.0, min(float(overlap_pixels), min(main_width, main_height) - 1.0))
        step_x = max(1.0, main_width - overlap)
        step_y = max(1.0, main_height - overlap)
        bounds = QRectF(self._image_rect)
        for column_offset, row_offset, image, image_path in frames:
            if column_offset == 0 and row_offset == 0:
                continue
            pixmap = QPixmap.fromImage(cv_to_qimage(image))
            if pixmap.isNull():
                continue
            item = QGraphicsPixmapItem(pixmap)
            item.setZValue(-20)
            item.setOpacity(max(0.05, min(1.0, float(opacity))))
            item.setToolTip(str(image_path))
            scale_x = main_width / max(1, pixmap.width())
            scale_y = main_height / max(1, pixmap.height())
            item.setTransform(QTransform.fromScale(scale_x, scale_y))
            item.setPos(float(column_offset) * step_x, float(row_offset) * step_y)
            self.addItem(item)
            self._neighbor_frame_items.append(item)
            self._neighbor_frame_paths[item] = str(image_path)
            bounds = bounds.united(QRectF(item.pos().x(), item.pos().y(), main_width, main_height))
        self._neighbor_grid_bounds = bounds if self._neighbor_frame_items else None
        self._update_scene_rect()

    def neighbor_frame_path_at(self, scene_pos: QPointF) -> str | None:
        for item in self.items(scene_pos):
            if isinstance(item, QGraphicsPixmapItem) and item in self._neighbor_frame_paths:
                return self._neighbor_frame_paths[item]
        return None

    def set_debug_candidates(self, candidates: list[object]) -> None:
        for item in self._debug_candidate_items:
            self.removeItem(item)
        self._debug_candidate_items.clear()
        _ = candidates

    def set_metal_overlays(
        self,
        layers: dict[str, list[PolygonData]],
        visibility: dict[str, bool],
    ) -> None:
        for item in self._metal_overlay_items:
            self.removeItem(item)
        self._metal_overlay_items.clear()
        layer_styles: list[tuple[str, str, bool]] = [
            ("rejected", "#EF4444", False),
            ("suspicious", "#EAB308", False),
            ("border", "#3B82F6", False),
            ("wide_pairs_suspicious", "#EAB308", True),
            ("wide_pairs_rejected", "#DC2626", True),
        ]
        z = 2.2
        _ru = self._ui_language == "ru"
        _layer_tip = {
            "rejected": "Отклонён" if _ru else "Rejected",
            "suspicious": "Сомнительный" if _ru else "Suspicious",
            "border": "У границы кадра" if _ru else "Border touch",
            "wide_pairs_suspicious": "Широкий проводник (сомнительно)" if _ru else "Wide trace (suspicious)",
            "wide_pairs_rejected": "Широкий проводник (отклонён)" if _ru else "Wide trace (rejected)",
        }
        for key, color_hex, dashed in layer_styles:
            if not visibility.get(key, False):
                continue
            for poly in layers.get(key) or []:
                path_item = QGraphicsPathItem(_display_path_for_polygon(poly))
                path_item.setZValue(z)
                c = QColor(color_hex)
                pen = QPen(c, 1.75)
                pen.setCosmetic(True)
                if dashed:
                    pen.setStyle(Qt.PenStyle.DashLine)
                path_item.setPen(pen)
                path_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                reason = str(getattr(poly, "reject_reason", "") or "")
                cap = _layer_tip.get(key, key)
                tip_lines = [cap]
                if reason.strip():
                    tip_lines.append(reason)
                path_item.setToolTip("\n".join(tip_lines))
                path_item.setData(int(Qt.ItemDataRole.UserRole), key)
                path_item.setData(int(Qt.ItemDataRole.UserRole) + 1, reason)
                self.addItem(path_item)
                self._metal_overlay_items.append(path_item)

    def metal_overlay_pick(self, scene_pos: QPointF) -> tuple[str, str] | None:
        """Return ``(layer_key, reject_reason)`` for the topmost metal overlay under ``scene_pos``."""

        for item in self.items(scene_pos):
            if item in self._metal_overlay_items:
                layer = item.data(int(Qt.ItemDataRole.UserRole))
                if layer is None:
                    continue
                reason = item.data(int(Qt.ItemDataRole.UserRole) + 1)
                return (str(layer), str(reason or ""))
        return None

    def set_gradient_overlay(self, image, opacity: float = 0.45) -> None:
        if image is None:
            self._gradient_overlay_item.setPixmap(QPixmap())
            self._gradient_overlay_item.hide()
            return
        pixmap = QPixmap.fromImage(cv_to_qimage(image))
        if pixmap.isNull():
            self._gradient_overlay_item.setPixmap(QPixmap())
            self._gradient_overlay_item.hide()
            return
        self._gradient_overlay_item.setPixmap(pixmap)
        self._gradient_overlay_item.setOpacity(max(0.0, min(1.0, float(opacity))))
        self._gradient_overlay_item.setPos(0.0, 0.0)
        self._gradient_overlay_item.show()

    def clear_gradient_overlay(self) -> None:
        self._gradient_overlay_item.setPixmap(QPixmap())
        self._gradient_overlay_item.hide()

    def set_gradient_overlay_opacity(self, opacity: float) -> None:
        self._gradient_overlay_item.setOpacity(max(0.0, min(1.0, float(opacity))))

    def _update_main_frame(self) -> None:
        path = QPainterPath()
        path.addRect(self._image_rect)
        self._main_frame_item.setPath(path)

    def _update_scene_rect(self) -> None:
        rect = QRectF(self._image_rect)
        if self._neighbor_grid_bounds is not None:
            rect = rect.united(self._neighbor_grid_bounds)
        self.setSceneRect(rect)

    def set_ui_language(self, language: str | None) -> None:
        self._ui_language = active_language(language)

    def warn_invalid_polygon_geometry(self) -> None:
        self.logRequested.emit(tr("polygon_invalid_geometry_log", language=self._ui_language))

    def set_display_settings(self, settings: DisplaySettings) -> None:
        self._display_settings = settings
        self._refresh_all_items()

    def set_random_object_colors_enabled(self, enabled: bool) -> None:
        self._random_object_colors_enabled = bool(enabled)
        self._refresh_all_items()

    def set_extra_layers(self, layers: list[dict[str, object]]) -> None:
        for item in self._extra_layer_items:
            self.removeItem(item)
        self._extra_layer_items.clear()
        for layer in layers:
            if not bool(layer.get("visible", True)):
                continue
            pixmap = layer.get("pixmap")
            if not isinstance(pixmap, QPixmap) or pixmap.isNull():
                continue
            dx = float(layer.get("dx", 0.0) or 0.0)
            dy = float(layer.get("dy", 0.0) or 0.0)
            opacity = max(0.0, min(1.0, float(layer.get("opacity", 0.35) or 0.35)))
            item = QGraphicsPixmapItem(pixmap)
            item.setZValue(0.8)
            item.setOpacity(opacity)
            item.setPos(dx, dy)
            item.setToolTip(str(layer.get("name", "")))
            self.addItem(item)
            self._extra_layer_items.append(item)

    def set_polygon_overlays_visible(self, visible: bool) -> None:
        self._polygon_overlays_visible = bool(visible)
        for polygon_id, item in self._polygon_items.items():
            poly = self._polygons[polygon_id]
            cat = str(getattr(poly, "category", "") or "")
            vis = self._polygon_category_visible.get(cat, True)
            item.setVisible(self._polygon_overlays_visible and vis)
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
        self._hover_conductor_polygon_id = None
        self._selected_polygon_id = None
        self._selected_polygon_ids.clear()
        self._next_polygon_id = 1
        for polygon in polygons:
            self._add_polygon_internal(polygon.clone(), emit_signal=False, refresh=False)
        if polygons:
            self._next_polygon_id = max(polygon.id for polygon in polygons) + 1
            self._selected_polygon_id = polygons[0].id
            self._selected_polygon_ids = {polygons[0].id}
        self._refresh_all_items()
        self.polygonsChanged.emit()
        self.activePolygonChanged.emit(self._selected_polygon_id)

    def sync_conductor_hover_highlight(self, scene_pos: QPointF) -> None:
        if not self._polygon_overlays_visible:
            self._set_hover_conductor_polygon_id(None)
            return
        underneath = self.polygon_at(scene_pos)
        target_id = resolve_conductor_hover_target_id(self._polygons, underneath)
        self._set_hover_conductor_polygon_id(target_id)

    def clear_conductor_hover_highlight(self) -> None:
        self._set_hover_conductor_polygon_id(None)

    def _set_hover_conductor_polygon_id(self, conductor_id: int | None) -> None:
        if conductor_id is not None and conductor_id not in self._polygons:
            conductor_id = None
        if conductor_id == self._hover_conductor_polygon_id:
            return
        self._hover_conductor_polygon_id = conductor_id
        self._refresh_all_items()

    def selected_polygon_id(self) -> int | None:
        return self._selected_polygon_id

    def select_polygon(self, polygon_id: int | None, *, additive: bool = False) -> None:
        if polygon_id is not None and polygon_id not in self._polygons:
            polygon_id = None
        if polygon_id is None:
            if not additive:
                self._selected_polygon_ids.clear()
            self._selected_polygon_id = None
        elif additive:
            if polygon_id in self._selected_polygon_ids:
                self._selected_polygon_ids.remove(polygon_id)
                self._selected_polygon_id = next(iter(sorted(self._selected_polygon_ids)), None)
            else:
                self._selected_polygon_ids.add(polygon_id)
                self._selected_polygon_id = polygon_id
        else:
            self._selected_polygon_ids = {polygon_id}
            self._selected_polygon_id = polygon_id
        self._refresh_all_items()
        self.activePolygonChanged.emit(self._selected_polygon_id)

    def select_polygons(self, polygon_ids: list[int]) -> None:
        selected_ids = {polygon_id for polygon_id in polygon_ids if polygon_id in self._polygons}
        self._selected_polygon_ids = selected_ids
        self._selected_polygon_id = min(selected_ids) if selected_ids else None
        self._refresh_all_items()
        self.activePolygonChanged.emit(self._selected_polygon_id)

    def select_polygons_in_rect(self, rect: QRectF, *, additive: bool = False) -> None:
        normalized = rect.normalized()
        if normalized.width() <= 0.0 or normalized.height() <= 0.0:
            if not additive:
                self.select_polygon(None)
            return
        selected_ids = {
            polygon_id
            for polygon_id, polygon in self._polygons.items()
            if _polygon_data_rect(polygon).intersects(normalized)
        }
        if additive:
            selected_ids.update(self._selected_polygon_ids)
        self.select_polygons(sorted(selected_ids))

    def polygon_snapshot(self, polygon_id: int | None) -> PolygonData | None:
        if polygon_id is None or polygon_id not in self._polygons:
            return None
        return self._polygons[polygon_id].clone()

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
        candidate_ids.extend(
            polygon_id for polygon_id in sorted(self._selected_polygon_ids) if polygon_id != self._selected_polygon_id
        )
        candidate_ids.extend(
            polygon_id for polygon_id in sorted(self._polygons) if polygon_id not in self._selected_polygon_ids
        )
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
        target_ids = [polygon_id] if polygon_id is not None else sorted(self._selected_polygon_ids)
        if not target_ids and self._selected_polygon_id is not None:
            target_ids = [self._selected_polygon_id]
        target_polygons = [self._polygons[target_id].clone() for target_id in target_ids if target_id in self._polygons]
        if not target_polygons:
            return False
        self.undo_stack.beginMacro("Delete polygons")
        try:
            for target_polygon in target_polygons:
                self.undo_stack.push(DeletePolygonCommand(self, target_polygon))
        finally:
            self.undo_stack.endMacro()
        return True

    def selected_polygons(self) -> list[PolygonData]:
        return [
            self._polygons[polygon_id].clone()
            for polygon_id in sorted(self._selected_polygon_ids)
            if polygon_id in self._polygons
        ]

    def add_cloned_polygons_at(
        self,
        polygons: list[PolygonData],
        source_anchor: QPointF,
        target_anchor: QPointF,
    ) -> list[int]:
        if not polygons:
            return []
        dx = target_anchor.x() - source_anchor.x()
        dy = target_anchor.y() - source_anchor.y()
        id_map: dict[int, int] = {}
        new_polygons: list[PolygonData] = []
        for polygon in polygons:
            shifted_points = [(float(x) + dx, float(y) + dy) for x, y in polygon.points]
            area, perimeter, bbox = compute_polygon_metrics(shifted_points)
            new_id = self._next_polygon_id
            self._next_polygon_id += 1
            id_map[polygon.id] = new_id
            new_polygons.append(
                PolygonData(
                    id=new_id,
                    points=shifted_points,
                    is_hole=polygon.is_hole,
                    parent_id=polygon.parent_id,
                    category=polygon.category,
                    shape_hint=polygon.shape_hint,
                    area=area,
                    perimeter=perimeter,
                    bbox=bbox,
                )
            )
        for polygon in new_polygons:
            polygon.parent_id = None if polygon.parent_id is None else id_map.get(polygon.parent_id)
        self.undo_stack.beginMacro("Paste polygons")
        try:
            for polygon in new_polygons:
                self.undo_stack.push(AddPolygonCommand(self, polygon))
        finally:
            self.undo_stack.endMacro()
        if new_polygons:
            self.select_polygons([polygon.id for polygon in new_polygons])
        return [polygon.id for polygon in new_polygons]

    def add_vertex_at(self, polygon_id: int, scene_pos: QPointF) -> bool:
        if polygon_id not in self._polygons:
            return False
        insert_index = self._nearest_segment_insert_index(polygon_id, scene_pos)
        new_point = (float(scene_pos.x()), float(scene_pos.y()))
        points = self.polygon_points(polygon_id)
        insert_at = max(0, min(len(points), insert_index))
        trial = list(points)
        trial.insert(insert_at, new_point)
        if not is_valid_closed_polygon_ring(trial):
            self.warn_invalid_polygon_geometry()
            return False
        self.undo_stack.push(AddVertexCommand(self, polygon_id, insert_index, new_point))
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
        if (
            self._pending_points
            and hypot(point[0] - self._pending_points[-1][0], point[1] - self._pending_points[-1][1]) < 1.0
        ):
            return
        trial = [*self._pending_points, point]
        if not is_valid_open_polyline_last_edge(trial):
            self.warn_invalid_polygon_geometry()
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
        if not is_valid_closed_polygon_ring(self._pending_points):
            self.warn_invalid_polygon_geometry()
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
        self._brush_cursor_item.setRect(
            QRectF(scene_pos.x() - radius, scene_pos.y() - radius, radius * 2.0, radius * 2.0)
        )
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
        if not is_valid_closed_polygon_ring(self._pending_points):
            self.warn_invalid_polygon_geometry()
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
                    self.undo_stack.push(
                        DeleteVertexCommand(self, polygon_id, vertex_index, polygon.points[vertex_index])
                    )
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
        shape_bbox = _bbox_from_points(points, padding=(round(thickness / 2.0) + 2) if thickness else 2)
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
        shape_bbox = _bbox_from_points(points, padding=(round(thickness / 2.0) + 2) if thickness else 2)
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
            conductor_hover_highlight = (
                self._hover_conductor_polygon_id is not None
                and polygon_id == self._hover_conductor_polygon_id
                and polygon_id not in self._selected_polygon_ids
            )
            item.update_from_polygon(
                self._polygons[polygon_id],
                self._display_settings,
                selected=polygon_id in self._selected_polygon_ids,
                cutout_polygons=self._cutout_polygons_for(polygon_id),
                custom_color=self._object_color_for(polygon_id),
                conductor_hover_highlight=conductor_hover_highlight,
            )
            poly = self._polygons[polygon_id]
            cat = str(getattr(poly, "category", "") or "")
            vis = self._polygon_category_visible.get(cat, True)
            item.setVisible(bool(vis) and self._polygon_overlays_visible)

    def set_polygon_category_visible(self, category: str, visible: bool) -> None:
        self._polygon_category_visible[str(category)] = bool(visible)
        for polygon_id, polygon in self._polygons.items():
            if str(getattr(polygon, "category", "") or "") != str(category):
                continue
            item = self._polygon_items.get(polygon_id)
            if item is not None:
                item.setVisible(bool(visible) and self._polygon_overlays_visible)

    def _object_color_for(self, polygon_id: int) -> str | None:
        if not self._random_object_colors_enabled:
            return None
        if polygon_id not in self._object_colors:
            self._object_colors[polygon_id] = _stable_object_color(polygon_id)
        return self._object_colors[polygon_id]

    def _cutout_polygons_for(self, polygon_id: int) -> list[PolygonData]:
        polygon = self._polygons.get(polygon_id)
        if polygon is None or polygon.is_hole:
            return []
        return [child.clone() for child in self._polygons.values() if child.parent_id == polygon_id and child.is_hole]

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
            pending.extend(child_id for child_id, polygon in self._polygons.items() if polygon.parent_id == current_id)
        return sorted(family_ids)

    def _assign_polygon_ids(
        self,
        polygons: list[PolygonData],
        replaced_ids: list[int],
        *,
        reserved_ids: set[int] | None = None,
    ) -> list[PolygonData]:
        reserved = reserved_ids or set()
        reusable_ids = [
            polygon_id
            for polygon_id in sorted(
                replaced_ids,
                key=lambda current_id: (
                    -abs(float(self._polygons[current_id].area)) if current_id in self._polygons else 0.0,
                    current_id,
                ),
            )
            if polygon_id not in reserved
        ]
        sorted_polygons = sorted(polygons, key=lambda polygon: -abs(float(polygon.area)))
        allocated_ids = self._allocate_polygon_ids(reusable_ids, len(sorted_polygons))
        id_map = {
            polygon.id: allocated_id for polygon, allocated_id in zip(sorted_polygons, allocated_ids, strict=False)
        }
        for polygon, allocated_id in zip(sorted_polygons, allocated_ids, strict=False):
            polygon.parent_id = None if polygon.parent_id is None else id_map.get(polygon.parent_id)
            polygon.id = allocated_id
        return sorted(sorted_polygons, key=lambda polygon: polygon.id)

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
        if self._hover_conductor_polygon_id == polygon_id:
            self._hover_conductor_polygon_id = None
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

    def _replace_polygon_points_internal(
        self, polygon_id: int, points: list[tuple[float, float]], emit_signal: bool = True
    ) -> None:
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
