from __future__ import annotations

from math import hypot

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QKeySequence, QPainter, QPainterPath, QPen, QShortcut, QUndoStack
from PyQt6.QtWidgets import QGraphicsPathItem, QGraphicsView

from ..application.processing import DisplaySettings
from ..commands import MovePolygonCommand, MoveVertexCommand
from ..domain import PolygonData
from .editor_scene import PolygonEditorScene
from .geometry import (
    _points_different,
    _polygon_points_different,
    _polygons_center,
    is_valid_closed_polygon_ring,
    _snap_to_45,
)
from .tools import BrushMode, DeleteVertexMode, EditorTool, PolygonCreateMode


class PolygonEditorView(QGraphicsView):
    polygonsEdited = pyqtSignal()
    activePolygonChanged = pyqtSignal(object)
    logRequested = pyqtSignal(str)
    imageClicked = pyqtSignal(float, float)
    imageRegionSelected = pyqtSignal(float, float, float, float)
    rulerMeasurementChanged = pyqtSignal(str)
    toolChanged = pyqtSignal(object)
    zoomChanged = pyqtSignal(float)
    neighborFrameActivated = pyqtSignal(str)
    viaDebugRequested = pyqtSignal(object)
    metalOverlayDetailRequested = pyqtSignal(str, str)
    middlePreviewHoldChanged = pyqtSignal(bool)

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
        self._image_region_selection_mode = False
        self._via_debug_inspection_enabled = False
        self._clipboard_polygons: list[PolygonData] = []
        self._clipboard_anchor = QPointF(0.0, 0.0)
        self._paste_mode = False
        self._paste_preview_items: list[QGraphicsPathItem] = []

        self._editor_scene.polygonsChanged.connect(self.polygonsEdited.emit)
        self._editor_scene.activePolygonChanged.connect(self.activePolygonChanged.emit)
        self._editor_scene.logRequested.connect(self.logRequested.emit)

        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self.redo)
        QShortcut(QKeySequence.StandardKey.Copy, self, activated=self.copy_selected)
        QShortcut(QKeySequence.StandardKey.Cut, self, activated=self.cut_selected)
        QShortcut(QKeySequence.StandardKey.Paste, self, activated=self.start_paste_mode)

    @property
    def undo_stack(self) -> QUndoStack:
        return self._editor_scene.undo_stack

    @property
    def current_tool(self) -> EditorTool:
        return self._tool

    def set_tool(self, tool: EditorTool) -> None:
        self._tool = tool
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag if tool == EditorTool.PAN else QGraphicsView.DragMode.NoDrag
        )
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

    def set_neighbor_frames(
        self,
        frames: list[tuple[int, int, object, str]],
        opacity: float,
        overlap_pixels: int = 0,
        show_main_frame: bool = True,
    ) -> None:
        self._editor_scene.set_neighbor_frames(frames, opacity, overlap_pixels, show_main_frame)

    def set_debug_candidates(self, candidates: list[object]) -> None:
        self._editor_scene.set_debug_candidates(candidates)

    def set_metal_overlays(self, layers: dict[str, list[PolygonData]], visibility: dict[str, bool]) -> None:
        self._editor_scene.set_metal_overlays(layers, visibility)

    def set_via_debug_inspection_enabled(self, enabled: bool) -> None:
        self._via_debug_inspection_enabled = bool(enabled)
        self._editor_scene.set_debug_candidates([])

    def zoom_factor(self) -> float:
        return max(1e-6, float(self.transform().m11()))

    def set_display_settings(self, settings: DisplaySettings) -> None:
        self._editor_scene.set_display_settings(settings)

    def set_random_object_colors_enabled(self, enabled: bool) -> None:
        self._editor_scene.set_random_object_colors_enabled(enabled)

    def set_extra_layers(self, layers: list[dict[str, object]]) -> None:
        self._editor_scene.set_extra_layers(layers)

    def set_gradient_overlay(self, image, opacity: float = 0.45) -> None:
        self._editor_scene.set_gradient_overlay(image, opacity)

    def clear_gradient_overlay(self) -> None:
        self._editor_scene.clear_gradient_overlay()

    def set_gradient_overlay_opacity(self, opacity: float) -> None:
        self._editor_scene.set_gradient_overlay_opacity(opacity)

    def set_polygon_category_visible(self, category: str, visible: bool) -> None:
        self._editor_scene.set_polygon_category_visible(category, visible)

    def set_ui_language(self, language: str | None) -> None:
        self._editor_scene.set_ui_language(language)

    def set_image_click_mode(self, enabled: bool) -> None:
        self._image_click_mode = bool(enabled)

    def set_image_region_selection_mode(self, enabled: bool) -> None:
        self._image_region_selection_mode = bool(enabled)
        if not enabled:
            self._editor_scene.clear_preview_rect()

    def fit_to_view(self) -> None:
        rect = self._editor_scene.main_image_rect()
        if rect.width() > 0 and rect.height() > 0:
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self.zoomChanged.emit(self.zoom_factor())

    def zoom_in(self) -> None:
        self.scale(1.15, 1.15)
        self.zoomChanged.emit(self.zoom_factor())

    def zoom_out(self) -> None:
        self.scale(1 / 1.15, 1 / 1.15)
        self.zoomChanged.emit(self.zoom_factor())

    def undo(self) -> None:
        self.undo_stack.undo()

    def redo(self) -> None:
        self.undo_stack.redo()

    def copy_selected(self) -> None:
        polygons = self._editor_scene.selected_polygons()
        if not polygons:
            return
        self._clipboard_polygons = [polygon.clone() for polygon in polygons]
        self._clipboard_anchor = _polygons_center(self._clipboard_polygons)

    def cut_selected(self) -> None:
        self.copy_selected()
        if self._clipboard_polygons:
            self._editor_scene.delete_polygon()

    def start_paste_mode(self) -> None:
        if not self._clipboard_polygons:
            return
        self._paste_mode = True
        self._update_paste_preview(self._last_pointer_scene_pos or self.mapToScene(self.viewport().rect().center()))

    def _clear_paste_preview(self) -> None:
        for item in self._paste_preview_items:
            if item.scene() is not None:
                self._editor_scene.removeItem(item)
        self._paste_preview_items.clear()

    def _exit_paste_mode(self) -> None:
        self._paste_mode = False
        self._clear_paste_preview()

    def _update_paste_preview(self, scene_pos: QPointF | None) -> None:
        self._clear_paste_preview()
        if not self._paste_mode or scene_pos is None:
            return
        dx = scene_pos.x() - self._clipboard_anchor.x()
        dy = scene_pos.y() - self._clipboard_anchor.y()
        pen = QPen(QColor("#38BDF8"), 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        brush = QColor("#38BDF8")
        brush.setAlpha(42)
        for polygon in self._clipboard_polygons:
            shifted = polygon.clone()
            shifted.points = [(x + dx, y + dy) for x, y in shifted.points]
            path = QPainterPath()
            if shifted.shape_hint == "box" or shifted.category == "via":
                x_values = [point[0] for point in shifted.points]
                y_values = [point[1] for point in shifted.points]
                path.addEllipse(
                    QRectF(min(x_values), min(y_values), max(x_values) - min(x_values), max(y_values) - min(y_values))
                )
            else:
                if shifted.points:
                    path.moveTo(shifted.points[0][0], shifted.points[0][1])
                    for x_coord, y_coord in shifted.points[1:]:
                        path.lineTo(x_coord, y_coord)
                    path.closeSubpath()
            item = QGraphicsPathItem(path)
            item.setZValue(40)
            item.setPen(pen)
            item.setBrush(QBrush(brush))
            self._editor_scene.addItem(item)
            self._paste_preview_items.append(item)

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
            self.zoomChanged.emit(self.zoom_factor())
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

        if self._paste_mode and event.button() == Qt.MouseButton.LeftButton:
            self._editor_scene.add_cloned_polygons_at(self._clipboard_polygons, self._clipboard_anchor, scene_pos)
            self._update_paste_preview(scene_pos)
            event.accept()
            return

        if self._image_region_selection_mode and event.button() == Qt.MouseButton.LeftButton:
            self._drag_kind = "image_region"
            self._drag_start_scene_pos = scene_pos
            self._editor_scene.set_preview_rect(scene_pos, scene_pos)
            event.accept()
            return

        if self._image_click_mode and event.button() == Qt.MouseButton.LeftButton:
            self.imageClicked.emit(scene_pos.x(), scene_pos.y())
            event.accept()
            return

        if self._tool == EditorTool.PAN:
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            metal_hit = self._editor_scene.metal_overlay_pick(scene_pos)
            if metal_hit is not None:
                self.metalOverlayDetailRequested.emit(metal_hit[0], metal_hit[1])
                event.accept()
                return

        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_button_hides_overlays = True
            self._editor_scene.set_polygon_overlays_visible(False)
            self.middlePreviewHoldChanged.emit(True)
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
            if (
                self._polygon_create_mode == PolygonCreateMode.RECTANGLE
                and event.button() == Qt.MouseButton.RightButton
            ):
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

        if self._tool == EditorTool.SELECT_AREA:
            self._drag_kind = "select_area"
            self._drag_start_scene_pos = scene_pos
            self._editor_scene.set_preview_rect(scene_pos, scene_pos)
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
        additive_selection = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        self._editor_scene.select_polygon(polygon_id, additive=additive_selection)
        if self._via_debug_inspection_enabled and polygon_id is not None:
            polygon = self._editor_scene.polygon_snapshot(polygon_id)
            if polygon is not None:
                self.viaDebugRequested.emit(polygon)
                event.accept()
                return
        if self._via_debug_inspection_enabled:
            event.accept()
            return
        if polygon_id is not None and event.modifiers() & Qt.KeyboardModifier.AltModifier:
            self._drag_kind = "polygon"
            self._drag_polygon_id = polygon_id
            self._drag_origin_points = self._editor_scene.polygon_points(polygon_id)
            self._drag_start_scene_pos = scene_pos
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        self._last_pointer_scene_pos = scene_pos
        self._update_tool_cursors()
        if self._paste_mode:
            self._editor_scene.clear_conductor_hover_highlight()
            self._update_paste_preview(scene_pos)
            event.accept()
            return
        self._editor_scene.sync_conductor_hover_highlight(scene_pos)
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
        if self._drag_kind == "select_area" and self._drag_start_scene_pos is not None:
            self._editor_scene.set_preview_rect(self._drag_start_scene_pos, scene_pos)
            event.accept()
            return
        if self._drag_kind == "image_region" and self._drag_start_scene_pos is not None:
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
        if (
            self._drag_kind == "polygon"
            and self._drag_polygon_id is not None
            and self._drag_origin_points is not None
            and self._drag_start_scene_pos is not None
        ):
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
            self.middlePreviewHoldChanged.emit(False)
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
                self._editor_scene.delete_vertices_in_rect(
                    QRectF(self._drag_start_scene_pos, self.mapToScene(event.position().toPoint()))
                )
                self._editor_scene.clear_preview_rect()
            elif self._drag_kind == "select_area" and self._drag_start_scene_pos is not None:
                release_pos = self.mapToScene(event.position().toPoint())
                rect = QRectF(self._drag_start_scene_pos, release_pos).normalized()
                self._editor_scene.clear_preview_rect()
                if rect.width() < self._scene_tolerance(3) and rect.height() < self._scene_tolerance(3):
                    polygon_id = self._editor_scene.polygon_at(release_pos)
                    self._editor_scene.select_polygon(
                        polygon_id, additive=bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    )
                else:
                    self._editor_scene.select_polygons_in_rect(
                        rect,
                        additive=bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier),
                    )
            elif self._drag_kind == "image_region" and self._drag_start_scene_pos is not None:
                rect = QRectF(self._drag_start_scene_pos, self.mapToScene(event.position().toPoint())).normalized()
                image_rect = self._editor_scene.main_image_rect()
                clipped = rect.intersected(image_rect)
                self._editor_scene.clear_preview_rect()
                if clipped.width() >= 2.0 and clipped.height() >= 2.0:
                    self.imageRegionSelected.emit(clipped.x(), clipped.y(), clipped.width(), clipped.height())
            elif (
                self._drag_kind == "vertex"
                and self._drag_polygon_id is not None
                and self._drag_vertex_index is not None
                and self._drag_origin_points is not None
            ):
                new_points = self._editor_scene.polygon_points(self._drag_polygon_id)
                old_point = self._drag_origin_points[self._drag_vertex_index]
                new_point = new_points[self._drag_vertex_index]
                if _points_different(old_point, new_point):
                    if not is_valid_closed_polygon_ring(new_points):
                        self._editor_scene.preview_vertex_move(
                            self._drag_polygon_id, self._drag_vertex_index, QPointF(old_point[0], old_point[1])
                        )
                        self._editor_scene.warn_invalid_polygon_geometry()
                    else:
                        self.undo_stack.push(
                            MoveVertexCommand(
                                self._editor_scene,
                                self._drag_polygon_id,
                                self._drag_vertex_index,
                                old_point,
                                new_point,
                            )
                        )
            elif (
                self._drag_kind == "polygon"
                and self._drag_polygon_id is not None
                and self._drag_origin_points is not None
            ):
                new_points = self._editor_scene.polygon_points(self._drag_polygon_id)
                if _polygon_points_different(self._drag_origin_points, new_points):
                    if not is_valid_closed_polygon_ring(new_points):
                        self._editor_scene.preview_polygon_move(self._drag_polygon_id, self._drag_origin_points)
                        self._editor_scene.warn_invalid_polygon_geometry()
                    else:
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
        if event.button() == Qt.MouseButton.LeftButton:
            neighbor_path = self._editor_scene.neighbor_frame_path_at(self.mapToScene(event.position().toPoint()))
            if neighbor_path:
                self.neighborFrameActivated.emit(neighbor_path)
                event.accept()
                return
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
            self._editor_scene.clear_preview_rect()
            self._exit_paste_mode()
            if self._tool in (EditorTool.SELECT, EditorTool.SELECT_AREA):
                self._editor_scene.select_polygon(None)
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
        if (
            event.key() == Qt.Key.Key_Shift
            and self._drag_kind == "ruler"
            and self._drag_start_scene_pos is not None
            and self._last_pointer_scene_pos is not None
        ):
            target_pos = self._ruler_target(self._drag_start_scene_pos, self._last_pointer_scene_pos, event.modifiers())
            measurement_text = self._format_ruler_measurement(self._drag_start_scene_pos, target_pos)
            self._editor_scene.set_measurement(self._drag_start_scene_pos, target_pos, measurement_text)
            self.rulerMeasurementChanged.emit(measurement_text)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if (
            event.key() == Qt.Key.Key_Shift
            and self._drag_kind == "ruler"
            and self._drag_start_scene_pos is not None
            and self._last_pointer_scene_pos is not None
        ):
            target_pos = self._ruler_target(self._drag_start_scene_pos, self._last_pointer_scene_pos, event.modifiers())
            measurement_text = self._format_ruler_measurement(self._drag_start_scene_pos, target_pos)
            self._editor_scene.set_measurement(self._drag_start_scene_pos, target_pos, measurement_text)
            self.rulerMeasurementChanged.emit(measurement_text)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._editor_scene.clear_conductor_hover_highlight()
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

    def _ruler_target(self, start: QPointF, target: QPointF, modifiers: Qt.KeyboardModifier) -> QPointF:
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            return _snap_to_45(start, target)
        return QPointF(target)

    def _format_ruler_measurement(self, start: QPointF, end: QPointF) -> str:
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        distance = hypot(dx, dy)
        return f"L={distance:.1f}px, dX={dx:.1f}, dY={dy:.1f}"
