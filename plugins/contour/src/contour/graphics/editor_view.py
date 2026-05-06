from __future__ import annotations

from math import hypot

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QGuiApplication,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QShortcut,
    QUndoStack,
)
from PyQt6.QtWidgets import QGraphicsPathItem, QGraphicsView

from ..application.processing import DisplaySettings
from ..application.vector_geometry_postprocess import (
    VectorGeometrySettings,
    apply_polygon_points_to_clone,
    apply_vertex_position_to_clone,
    postprocess_changed_polygon_only,
    resolve_focus_id_after_geometry_pass,
)
from ..commands import ReplacePolygonSetCommand
from ..domain import PolygonData
from .editor_hotkeys import tool_shortcut_sequence
from .editor_scene import PolygonEditorScene
from .geometry import (
    _points_different,
    _polygon_points_different,
    _polygons_center,
    _snap_to_45,
    is_valid_closed_polygon_ring,
)
from .tool_mode_logic import effective_polygon_create_mode, normalize_editor_tool
from .tools import BrushMode, DeleteVertexMode, EditorTool, PolygonCreateMode
from .viewport_navigation import (
    polygon_overlay_visibility_after_space_toggle,
    viewport_scroll_correction_after_scale_reanchor,
)


class PolygonEditorView(QGraphicsView):
    polygonsEdited = pyqtSignal()
    activePolygonChanged = pyqtSignal(object)
    logRequested = pyqtSignal(str)
    imageClicked = pyqtSignal(float, float)
    imageRegionSelected = pyqtSignal(float, float, float, float)
    rulerMeasurementChanged = pyqtSignal(str)
    toolChanged = pyqtSignal(object)
    effectivePolygonCreateModeChanged = pyqtSignal(object)
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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._tool = EditorTool.SELECT
        self._polygon_create_mode = PolygonCreateMode.POINTS
        self._brush_mode = BrushMode.FREEFORM
        self._brush_thickness = 12.0
        self._via_width = 12.0
        self._via_height = 12.0
        self._delete_vertex_mode = DeleteVertexMode.SINGLE
        self._select_press_polygon_id: int | None = None
        self._select_press_start: QPointF | None = None
        self._drag_kind: str | None = None
        self._drag_polygon_id: int | None = None
        self._drag_vertex_index: int | None = None
        self._drag_origin_points: list[tuple[float, float]] | None = None
        self._drag_start_scene_pos: QPointF | None = None
        self._last_pointer_scene_pos: QPointF | None = None
        self._drag_erases = False
        self._pending_polygon_erases: bool | None = None
        self._middle_pan_active = False
        self._middle_pan_last_viewport: QPointF | None = None
        self._vectors_hidden_via_space_toggle = False
        self._last_pointer_viewport_pos: QPointF | None = None
        self._image_click_mode = False
        self._image_region_selection_mode = False
        self._via_debug_inspection_enabled = False
        self._clipboard_polygons: list[PolygonData] = []
        self._clipboard_anchor = QPointF(0.0, 0.0)
        self._paste_mode = False
        self._paste_preview_items: list[QGraphicsPathItem] = []
        self._vector_geometry_settings = VectorGeometrySettings()
        self._drag_polygons_snapshot: list[PolygonData] | None = None

        self._editor_scene.polygonsChanged.connect(self.polygonsEdited.emit)
        self._editor_scene.activePolygonChanged.connect(self.activePolygonChanged.emit)
        self._editor_scene.logRequested.connect(self.logRequested.emit)

        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self.redo)
        QShortcut(QKeySequence.StandardKey.Copy, self, activated=self.copy_selected)
        QShortcut(QKeySequence.StandardKey.Cut, self, activated=self.cut_selected)
        QShortcut(QKeySequence.StandardKey.Paste, self, activated=self.start_paste_mode)

        for tool in EditorTool:
            sequence = tool_shortcut_sequence(tool)
            if sequence is None:
                continue
            shortcut = QShortcut(sequence, self)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(lambda t=tool: self.set_tool(t))

    @property
    def undo_stack(self) -> QUndoStack:
        return self._editor_scene.undo_stack

    @property
    def current_tool(self) -> EditorTool:
        return self._tool

    def set_tool(self, tool: EditorTool) -> None:
        tool = normalize_editor_tool(tool)
        self._tool = tool
        self._select_press_polygon_id = None
        self._select_press_start = None
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
        self._emit_effective_polygon_create_mode_changed()

    def set_polygon_create_mode(self, mode: PolygonCreateMode) -> None:
        self._polygon_create_mode = mode
        self._editor_scene.cancel_pending_polygon()
        self._pending_polygon_erases = None
        self._emit_effective_polygon_create_mode_changed()

    def set_brush_mode(self, mode: BrushMode) -> None:
        # Stamp modes were removed from UI; coerce legacy/persisted values.
        if mode not in (BrushMode.FREEFORM, BrushMode.ANGLED):
            mode = BrushMode.FREEFORM
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

    def set_vector_geometry_settings(self, settings: VectorGeometrySettings | None) -> None:
        self._vector_geometry_settings = settings if settings is not None else VectorGeometrySettings()
        self._editor_scene.set_vector_geometry_settings(settings)

    def set_delete_vertex_mode(self, mode: DeleteVertexMode) -> None:
        self._delete_vertex_mode = mode
        self._editor_scene.clear_preview_rect()

    def _effective_polygon_create_mode(self) -> PolygonCreateMode:
        return effective_polygon_create_mode(
            tool=self._tool,
            base=self._polygon_create_mode,
            shift_held=bool(QGuiApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier),
            has_pending_polygon=self._editor_scene.has_pending_polygon(),
        )

    def effective_polygon_create_mode(self) -> PolygonCreateMode:
        """Polygon draw mode including Shift override (read-only, for UI/tests)."""
        return self._effective_polygon_create_mode()

    def _emit_effective_polygon_create_mode_changed(self) -> None:
        self.effectivePolygonCreateModeChanged.emit(self._effective_polygon_create_mode())

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
        self._apply_zoom_at_viewport_pixel(self._zoom_focus_viewport_pixel(), 1.15)
        self._update_tool_cursors()
        self.zoomChanged.emit(self.zoom_factor())

    def zoom_out(self) -> None:
        self._apply_zoom_at_viewport_pixel(self._zoom_focus_viewport_pixel(), 1.0 / 1.15)
        self._update_tool_cursors()
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
        # Coordinates are viewport-local (see QGraphicsView::wheelEvent).
        viewport_point = event.position().toPoint()
        delta = event.angleDelta()
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            if delta.y() == 0:
                event.accept()
                return
            factor = 1.15 ** (delta.y() / 120.0)
            self._apply_zoom_at_viewport_pixel(viewport_point, factor)
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
        if event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ):
            self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._last_pointer_viewport_pos = QPointF(event.position())
        scene_pos = self.mapToScene(event.position().toPoint())
        self._last_pointer_scene_pos = scene_pos
        tolerance = self._scene_tolerance(8)

        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_pan_active = True
            self._middle_pan_last_viewport = QPointF(event.position())
            self.middlePreviewHoldChanged.emit(True)
            event.accept()
            return

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

        if self._tool == EditorTool.ADD_POLYGON:
            create_mode = self._effective_polygon_create_mode()
            if create_mode == PolygonCreateMode.RECTANGLE and event.button() == Qt.MouseButton.LeftButton:
                self._drag_kind = "rect_polygon"
                self._drag_start_scene_pos = scene_pos
                self._drag_erases = False
                self._editor_scene.set_preview_rect(scene_pos, scene_pos)
                event.accept()
                return
            if create_mode == PolygonCreateMode.RECTANGLE and event.button() == Qt.MouseButton.RightButton:
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
                self._emit_effective_polygon_create_mode_changed()
                event.accept()
                return

        if self._tool == EditorTool.BRUSH and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._drag_erases = False
            self._drag_kind = "brush"
            self._drag_start_scene_pos = scene_pos
            if event.button() == Qt.MouseButton.RightButton:
                self._drag_erases = True

            self._editor_scene.start_pending_polygon(for_brush=True)
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

        if self._tool == EditorTool.SELECT:
            polygon_id = self._editor_scene.polygon_at(scene_pos)
            additive_selection = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if polygon_id is None:
                self._drag_kind = "select_area"
                self._drag_start_scene_pos = scene_pos
                self._select_press_polygon_id = None
                self._select_press_start = None
                self._editor_scene.set_preview_rect(scene_pos, scene_pos)
                event.accept()
                return
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
            self._select_press_polygon_id = polygon_id
            self._select_press_start = QPointF(scene_pos)
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
            if hit is None:
                # Practical fallback: users often click near the handle, not exactly on it.
                hit = self._editor_scene.vertex_at(scene_pos, self._scene_tolerance(12))
            if hit is not None:
                polygon_id, vertex_index = hit
                self._editor_scene.select_polygon(polygon_id)
                self._drag_kind = "vertex"
                self._drag_polygon_id = polygon_id
                self._drag_vertex_index = vertex_index
                self._drag_origin_points = self._editor_scene.polygon_points(polygon_id)
                self._drag_polygons_snapshot = self._editor_scene.get_polygons()
                self._drag_start_scene_pos = scene_pos
            event.accept()
            return

    def mouseMoveEvent(self, event) -> None:
        self._last_pointer_viewport_pos = QPointF(event.position())
        scene_pos = self.mapToScene(event.position().toPoint())
        self._last_pointer_scene_pos = scene_pos
        self._update_tool_cursors()
        if self._middle_pan_active and self._middle_pan_last_viewport is not None:
            cur = QPointF(event.position())
            dv = cur - self._middle_pan_last_viewport
            self.horizontalScrollBar().setValue(round(self.horizontalScrollBar().value() - dv.x()))
            self.verticalScrollBar().setValue(round(self.verticalScrollBar().value() - dv.y()))
            self._middle_pan_last_viewport = cur
            event.accept()
            return
        if self._paste_mode:
            self._editor_scene.clear_conductor_hover_highlight()
            self._update_paste_preview(scene_pos)
            event.accept()
            return
        self._editor_scene.sync_conductor_hover_highlight(scene_pos)
        if (
            self._tool == EditorTool.SELECT
            and self._select_press_polygon_id is not None
            and self._drag_kind is None
            and self._select_press_start is not None
            and bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        ):
            dx = scene_pos.x() - self._select_press_start.x()
            dy = scene_pos.y() - self._select_press_start.y()
            if hypot(dx, dy) >= self._scene_tolerance(4.0):
                self._drag_kind = "polygon"
                self._drag_polygon_id = self._select_press_polygon_id
                self._drag_origin_points = self._editor_scene.polygon_points(self._select_press_polygon_id)
                self._drag_polygons_snapshot = self._editor_scene.get_polygons()
                self._drag_start_scene_pos = QPointF(self._select_press_start)
                self._select_press_polygon_id = None
                self._select_press_start = None
        if self._tool == EditorTool.PAN:
            super().mouseMoveEvent(event)
            return
        if self._tool == EditorTool.ADD_POLYGON and (
            self._effective_polygon_create_mode() == PolygonCreateMode.POINTS
            or self._editor_scene.has_pending_polygon()
        ):
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
        if event.button() == Qt.MouseButton.MiddleButton and self._middle_pan_active:
            self._middle_pan_active = False
            self._middle_pan_last_viewport = None
            self.middlePreviewHoldChanged.emit(False)
            event.accept()
            return
        if self._tool == EditorTool.PAN:
            super().mouseReleaseEvent(event)
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._tool == EditorTool.SELECT
            and self._drag_kind is None
            and self._select_press_polygon_id is not None
        ):
            self._select_press_polygon_id = None
            self._select_press_start = None
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
                and self._drag_polygons_snapshot is not None
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
                        trial = apply_vertex_position_to_clone(
                            self._drag_polygons_snapshot,
                            self._drag_polygon_id,
                            self._drag_vertex_index,
                            new_point,
                        )
                        processed, _changed = postprocess_changed_polygon_only(
                            trial,
                            self._vector_geometry_settings,
                            polygon_id=self._drag_polygon_id,
                        )
                        if not _changed:
                            # Keep direct vertex move if local cleanup produced no effective topology update.
                            processed = trial
                        focus_id = resolve_focus_id_after_geometry_pass(
                            self._drag_polygons_snapshot,
                            self._drag_polygon_id,
                            processed,
                        )
                        self.undo_stack.push(
                            ReplacePolygonSetCommand(
                                self._editor_scene,
                                self._drag_polygons_snapshot,
                                processed,
                                "Move vertex",
                            )
                        )
                        self._editor_scene.select_polygon(focus_id)
            elif (
                self._drag_kind == "polygon"
                and self._drag_polygon_id is not None
                and self._drag_origin_points is not None
                and self._drag_polygons_snapshot is not None
            ):
                new_points = self._editor_scene.polygon_points(self._drag_polygon_id)
                if _polygon_points_different(self._drag_origin_points, new_points):
                    if not is_valid_closed_polygon_ring(new_points):
                        self._editor_scene.preview_polygon_move(self._drag_polygon_id, self._drag_origin_points)
                        self._editor_scene.warn_invalid_polygon_geometry()
                    else:
                        trial = apply_polygon_points_to_clone(
                            self._drag_polygons_snapshot,
                            self._drag_polygon_id,
                            new_points,
                        )
                        processed, _c = postprocess_changed_polygon_only(
                            trial,
                            self._vector_geometry_settings,
                            polygon_id=self._drag_polygon_id,
                        )
                        if not _c:
                            # Keep direct polygon move if local cleanup produced no effective topology update.
                            processed = trial
                        focus_id = resolve_focus_id_after_geometry_pass(
                            self._drag_polygons_snapshot,
                            self._drag_polygon_id,
                            processed,
                        )
                        self.undo_stack.push(
                            ReplacePolygonSetCommand(
                                self._editor_scene,
                                self._drag_polygons_snapshot,
                                processed,
                                "Move polygon",
                            )
                        )
                        self._editor_scene.select_polygon(focus_id)
            self._drag_kind = None
            self._drag_polygon_id = None
            self._drag_vertex_index = None
            self._drag_origin_points = None
            self._drag_start_scene_pos = None
            self._drag_polygons_snapshot = None
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
            and self._editor_scene.has_pending_polygon()
            and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)
        ):
            self._finish_pending_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        if (
            event.key() == Qt.Key.Key_Space
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and self.isEnabled()
            and (self.hasFocus() or self.viewport().hasFocus())
        ):
            if event.isAutoRepeat():
                event.accept()
                return
            new_hidden, overlays_visible = polygon_overlay_visibility_after_space_toggle(
                self._vectors_hidden_via_space_toggle
            )
            self._vectors_hidden_via_space_toggle = new_hidden
            self._editor_scene.set_polygon_overlays_visible(overlays_visible)
            event.accept()
            return
        if (
            event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return)
            and self._tool == EditorTool.ADD_POLYGON
            and self._editor_scene.has_pending_polygon()
        ):
            self._finish_pending_polygon()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self._editor_scene.cancel_pending_polygon()
            self._editor_scene.clear_measurement()
            self._editor_scene.clear_preview_rect()
            self._exit_paste_mode()
            if self._tool == EditorTool.SELECT:
                self._editor_scene.select_polygon(None)
            self._select_press_polygon_id = None
            self._select_press_start = None
            if self._tool == EditorTool.RULER:
                self.rulerMeasurementChanged.emit("")
            self._drag_kind = None
            self._drag_erases = False
            self._pending_polygon_erases = None
            self._update_tool_cursors()
            self._emit_effective_polygon_create_mode_changed()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Delete:
            self._editor_scene.delete_polygon()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Shift and self._tool == EditorTool.ADD_POLYGON and not event.isAutoRepeat():
            self._emit_effective_polygon_create_mode_changed()
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
        if event.key() == Qt.Key.Key_Shift and self._tool == EditorTool.ADD_POLYGON and not event.isAutoRepeat():
            self._emit_effective_polygon_create_mode_changed()
        super().keyReleaseEvent(event)

    def _zoom_focus_viewport_pixel(self) -> QPoint:
        if self._last_pointer_viewport_pos is not None:
            p = self._last_pointer_viewport_pos.toPoint()
            vr = self.viewport().rect()
            clamped = QPoint(p.x(), p.y())
            if not vr.contains(clamped):
                return vr.center()
            return clamped
        return self.viewport().rect().center()

    def _apply_zoom_at_viewport_pixel(self, viewport_pixel: QPoint, factor: float) -> None:
        if factor == 1.0 or factor <= 0:
            return
        scene_anchor = self.mapToScene(viewport_pixel)
        old_anchor = self.transformationAnchor()
        try:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
            self.scale(factor, factor)
        finally:
            self.setTransformationAnchor(old_anchor)
        vp_mapped = self.mapFromScene(scene_anchor)
        dh, dv = viewport_scroll_correction_after_scale_reanchor(
            (viewport_pixel.x(), viewport_pixel.y()),
            (vp_mapped.x(), vp_mapped.y()),
        )
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + dh)
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + dv)

    def leaveEvent(self, event) -> None:
        self._editor_scene.clear_conductor_hover_highlight()
        self._editor_scene.hide_tool_cursors()
        super().leaveEvent(event)

    def _scene_tolerance(self, pixels: float | int) -> float:
        px = max(1, int(round(pixels)))
        start = self.mapToScene(QPoint(0, 0))
        end = self.mapToScene(QPoint(px, 0))
        return max(1.0, abs(end.x() - start.x()))

    def _append_brush_point(self, scene_pos: QPointF) -> None:
        target = scene_pos
        if self._brush_mode == BrushMode.ANGLED:
            last_point = self._editor_scene.pending_last_point()
            if last_point is not None:
                target = _snap_to_45(last_point, scene_pos)
        self._editor_scene.append_brush_vertex(target, self._brush_thickness)

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
        self._emit_effective_polygon_create_mode_changed()

    def _ruler_target(self, start: QPointF, target: QPointF, modifiers: Qt.KeyboardModifier) -> QPointF:
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            return _snap_to_45(start, target)
        return QPointF(target)

    def _format_ruler_measurement(self, start: QPointF, end: QPointF) -> str:
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        distance = hypot(dx, dy)
        return f"L={distance:.1f}px, dX={dx:.1f}, dY={dy:.1f}"
