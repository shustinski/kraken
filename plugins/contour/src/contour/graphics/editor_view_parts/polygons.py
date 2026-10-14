"""editor_view: polygons."""
from __future__ import annotations
from PyQt6.QtCore import (
    QPoint,
    QPointF,
    QRectF,
)
from PyQt6.QtGui import (
    QPixmap,
    QTransform,
)
from ...application.vector_geometry_postprocess import (
    VectorGeometrySettings,
)
from contour.domain import PolygonData
from contour.graphics.tool_mode_logic import (
    apply_conductor_recognition_tool_lock,
    effective_polygon_create_mode,
)
from contour.graphics.tools import MIN_MANUAL_STROKE_WIDTH_PX, BrushMode, DeleteVertexMode, EditorTool, PolygonCreateMode


class PolygonsMixin:
    def available_tools(self) -> frozenset[EditorTool]:
        return self._available_tools

    def _refresh_available_tools_from_scene(self) -> None:
        available = apply_conductor_recognition_tool_lock(
            self._editor_scene.available_editor_tools(),
            enabled=self._conductor_recognition_mode,
        )
        changed = available != self._available_tools
        self._available_tools = available
        if self._tool not in available:
            self.set_tool(EditorTool.SELECT)
        if self._paste_mode and (
            self._conductor_recognition_mode
            or not self._editor_scene.can_add_polygon_set(self._clipboard_polygons)
        ):
            self._exit_paste_mode()
        if changed:
            self.availableToolsChanged.emit(available)

    def vector_edits_locked(self) -> bool:
        return bool(self._conductor_recognition_mode)

    def set_conductor_recognition_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._conductor_recognition_mode:
            self._refresh_available_tools_from_scene()
            self.availableToolsChanged.emit(self._available_tools)
            return
        self._conductor_recognition_mode = enabled
        if enabled:
            self._exit_paste_mode()
            self._editor_scene.cancel_pending_polygon()
            self._editor_scene.clear_preview_rect()
            self._select_press_polygon_id = None
            self._select_press_start = None
            self._drag_kind = None
        self._refresh_available_tools_from_scene()
        self.availableToolsChanged.emit(self._available_tools)

    def set_contact_recognition_mode(self, enabled: bool) -> None:
        self._contact_recognition_mode = bool(enabled)
        # Recognized contacts remain selectable on left click, but right click /
        # Delete must be able to turn one into negative heuristic feedback.
        self._editor_scene.set_protect_recognized_vias(False)

    def set_polygon_create_mode(self, mode: PolygonCreateMode) -> None:
        mode = PolygonCreateMode(mode)
        changed = mode != self._polygon_create_mode
        self._polygon_create_mode = mode
        self._cancel_rectangle_polygon()
        self._editor_scene.cancel_pending_polygon()
        self._pending_polygon_erases = None
        if changed:
            self.polygonCreateModeChanged.emit(mode)
        self._emit_effective_polygon_create_mode_changed()

    def set_brush_mode(self, mode: BrushMode) -> None:
        mode = BrushMode(mode)
        changed = mode != self._brush_mode
        self._brush_mode = mode
        self._editor_scene.cancel_pending_polygon()
        if changed:
            self.brushModeChanged.emit(mode)
        self._update_tool_cursors()

    def set_brush_thickness(self, thickness: float) -> None:
        self._brush_thickness = max(MIN_MANUAL_STROKE_WIDTH_PX, float(thickness))
        if self._tool == EditorTool.BRUSH:
            self._editor_scene.set_pending_path_width(self._brush_thickness, cosmetic=False)
        self._update_tool_cursors()

    def set_trace_width(self, width: float) -> None:
        self._trace_width = max(MIN_MANUAL_STROKE_WIDTH_PX, float(width))
        if self._tool == EditorTool.TRACE_PEN:
            self._editor_scene.set_pending_path_width(self._trace_width, cosmetic=False)
        self._update_tool_cursors()

    def set_trace_mode(self, mode: BrushMode) -> None:
        mode = BrushMode(mode)
        changed = mode != self._trace_mode
        self._trace_mode = mode
        if changed:
            self.traceModeChanged.emit(mode)
        if self._editor_scene.has_pending_polygon() and self._tool == EditorTool.TRACE_PEN:
            self._editor_scene.cancel_pending_polygon()
            self._pending_polygon_erases = None

    def set_via_size(self, width: float, height: float) -> None:
        self._via_width = max(1.0, float(width))
        self._via_height = max(1.0, float(height))

    def set_minimum_contact_distance(self, distance: float) -> None:
        self._editor_scene.set_minimum_contact_distance(distance)
        self._update_tool_cursors()

    def set_vector_geometry_settings(self, settings: VectorGeometrySettings | None) -> None:
        self._vector_geometry_settings = settings if settings is not None else VectorGeometrySettings()
        self._editor_scene.set_vector_geometry_settings(settings)

    def set_delete_vertex_mode(self, mode: DeleteVertexMode) -> None:
        mode = DeleteVertexMode(mode)
        changed = mode != self._delete_vertex_mode
        self._delete_vertex_mode = mode
        self._editor_scene.clear_preview_rect()
        self._sync_all_editable_vertices_display()
        if changed:
            self.deleteVertexModeChanged.emit(mode)

    def _should_show_all_editable_vertices(self) -> bool:
        if self._tool in {EditorTool.ANTIALIAS, EditorTool.MOVE_VERTEX}:
            return True
        return self._tool == EditorTool.DELETE_VERTEX and self._delete_vertex_mode == DeleteVertexMode.SINGLE

    def _sync_all_editable_vertices_display(self) -> float:
        return self._editor_scene.set_show_all_editable_vertices(
            self._should_show_all_editable_vertices()
        )

    def _effective_polygon_create_mode(self) -> PolygonCreateMode:
        return effective_polygon_create_mode(
            tool=self._tool,
            base=self._polygon_create_mode,
            shift_held=False,
            has_pending_polygon=self._editor_scene.has_pending_polygon(),
        )

    def effective_polygon_create_mode(self) -> PolygonCreateMode:
        """Polygon draw mode including Shift override (read-only, for UI/tests)."""
        return self._effective_polygon_create_mode()

    def _emit_effective_polygon_create_mode_changed(self) -> None:
        self.effectivePolygonCreateModeChanged.emit(self._effective_polygon_create_mode())

    def _viewport_view_anchor(self) -> tuple[QPoint, QPointF, QTransform]:
        viewport = self._require_viewport()
        pixel = viewport.rect().center()
        return pixel, self.mapToScene(pixel), QTransform(self.transform())

    def _restore_viewport_view_anchor(
        self,
        pixel: QPoint,
        scene_anchor: QPointF,
        transform: QTransform,
    ) -> None:
        self.setTransform(transform)
        mapped = self.mapFromScene(scene_anchor)
        dx = round(mapped.x() - pixel.x())
        dy = round(mapped.y() - pixel.y())
        if dx or dy:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + dx)
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + dy)

    def set_image(self, image, *, preserve_view: bool = False) -> None:
        previous_rect = QRectF(self._editor_scene.sceneRect())
        view_anchor = self._viewport_view_anchor() if preserve_view else None
        self._editor_scene.set_image(image)
        previous_was_placeholder = previous_rect.width() <= 1.0 and previous_rect.height() <= 1.0
        if previous_was_placeholder:
            self.fit_to_view()
        elif view_anchor is not None:
            self._restore_viewport_view_anchor(*view_anchor)
        self._refresh_minimap_thumbnail()
        self._update_navigation_scene_rect()
        if view_anchor is not None:
            self._restore_viewport_view_anchor(*view_anchor)

    def set_image_pixmap(self, pixmap: QPixmap, *, preserve_view: bool = False) -> None:
        previous_rect = QRectF(self._editor_scene.sceneRect())
        view_anchor = self._viewport_view_anchor() if preserve_view else None
        self._editor_scene.set_image_pixmap(pixmap)
        previous_was_placeholder = previous_rect.width() <= 1.0 and previous_rect.height() <= 1.0
        if previous_was_placeholder:
            self.fit_to_view()
        elif view_anchor is not None:
            self._restore_viewport_view_anchor(*view_anchor)
        self._refresh_minimap_thumbnail()
        self._update_navigation_scene_rect()
        if view_anchor is not None:
            self._restore_viewport_view_anchor(*view_anchor)

    def set_polygons(
        self,
        polygons: list[PolygonData],
        *,
        emit_signal: bool = True,
        repair_reasons: dict[int, list[str]] | None = None,
        scan_repair: bool = True,
        defer_hit_paths: bool = False,
        defer_object_colors: bool = False,
    ) -> None:
        self._editor_scene.set_polygons(
            polygons,
            emit_signal=emit_signal,
            repair_reasons=repair_reasons,
            scan_repair=scan_repair,
            defer_hit_paths=defer_hit_paths,
            defer_object_colors=defer_object_colors,
        )
        self._refresh_available_tools_from_scene()
