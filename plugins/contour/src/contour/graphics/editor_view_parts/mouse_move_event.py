"""editor_view: mouse move event."""
from __future__ import annotations
from math import hypot
from time import perf_counter
from PyQt6.QtCore import (
    QPointF,
    Qt,
)
from PyQt6.QtGui import (
    QMouseEvent,
)
from contour.graphics.geometry import (
    _snap_to_45,
)
from contour.graphics.tools import BrushMode, EditorTool, PolygonCreateMode


class MouseMoveEventMixin:
    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        move_event_started_at = perf_counter()
        viewport_pixel = self._require_viewport().mapFrom(self, event.position().toPoint())
        self._last_pointer_viewport_pos = QPointF(viewport_pixel)
        scene_pos = self.mapToScene(self._viewport_to_view_point(viewport_pixel))
        if not (self._middle_pan_active and self._drag_kind == "brush"):
            self._last_pointer_scene_pos = scene_pos
        self._update_tool_cursors()
        if self._middle_pan_active and self._middle_pan_last_viewport is not None:
            cur = QPointF(viewport_pixel)
            dv = cur - self._middle_pan_last_viewport
            h_scroll = self.horizontalScrollBar()
            v_scroll = self.verticalScrollBar()
            if h_scroll is not None:
                h_scroll.setValue(round(h_scroll.value() - dv.x()))
            if v_scroll is not None:
                v_scroll.setValue(round(v_scroll.value() - dv.y()))
            self._middle_pan_last_viewport = cur
            event.accept()
            return
        if self._paste_mode:
            self._editor_scene.clear_conductor_hover_highlight()
            self._update_paste_preview(scene_pos)
            event.accept()
            return
        brush_drag_active = self._drag_kind == "brush"
        trace_drag_active = self._drag_kind == "trace"
        selection_drag_active = self._drag_kind == "select_area"
        polygon_drag_active = self._drag_kind == "polygon"
        zoom_active = self._zoom_animation_timer.isActive()
        if (
            brush_drag_active
            or trace_drag_active
            or selection_drag_active
            or polygon_drag_active
            or zoom_active
        ):
            self._editor_scene.clear_conductor_hover_highlight()
        else:
            self._editor_scene.sync_conductor_hover_highlight(scene_pos)
        if self._tool == EditorTool.ANTIALIAS and self._drag_kind is None:
            self._editor_scene.sync_vertex_preview(scene_pos)
        else:
            self._editor_scene.clear_vertex_preview()
        if self._tool == EditorTool.MOVE_VERTEX and self._drag_kind is None and not zoom_active and not self._middle_pan_active:
            self._editor_scene.sync_move_target_preview(
                scene_pos,
                vertex_tolerance=self._scene_tolerance(8),
                edge_tolerance=self._scene_tolerance(10),
            )
        elif not zoom_active and not self._middle_pan_active:
            self._editor_scene.clear_move_target_preview()
        if (
            self._tool == EditorTool.SELECT
            and not self.vector_edits_locked()
            and self._select_press_polygon_id is not None
            and self._drag_kind is None
            and self._select_press_start is not None
            and bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        ):
            dx = scene_pos.x() - self._select_press_start.x()
            dy = scene_pos.y() - self._select_press_start.y()
            if hypot(dx, dy) >= self._scene_tolerance(4.0):
                self._start_contact_drag_profile(self._select_press_polygon_id)
                self._drag_kind = "polygon"
                self._drag_polygon_id = self._select_press_polygon_id
                self._drag_polygon_is_contact = self._editor_scene.polygon_is_contact(
                    self._select_press_polygon_id
                )
                self._drag_origin_points = self._editor_scene.polygon_points(self._select_press_polygon_id)
                self._drag_preview_points = list(self._drag_origin_points)
                self._editor_scene.begin_polygon_edit_preview(self._select_press_polygon_id)
                self._drag_polygons_snapshot = (
                    None
                    if self._drag_polygon_is_contact
                    else self._editor_scene.get_polygons()
                )
                self._drag_start_scene_pos = QPointF(self._select_press_start)
                self._select_press_polygon_id = None
                self._select_press_start = None
        if self._tool == EditorTool.PAN:
            super().mouseMoveEvent(event)
            return
        if self._drag_kind == "rect_polygon" and self._drag_start_scene_pos is not None:
            self._editor_scene.set_preview_rect(self._drag_start_scene_pos, scene_pos)
            event.accept()
            return
        if self._tool == EditorTool.ADD_POLYGON and (
            self._effective_polygon_create_mode() == PolygonCreateMode.POINTS
            or self._editor_scene.has_pending_polygon()
        ):
            self._editor_scene.update_pending_cursor(scene_pos)
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
            self._editor_scene.preview_delete_vertices_in_rect(self._drag_start_scene_pos, scene_pos)
            event.accept()
            return
        if self._drag_kind == "select_area" and self._drag_start_scene_pos is not None:
            self._editor_scene.set_preview_rect(self._drag_start_scene_pos, scene_pos)
            event.accept()
            return
        if self._drag_kind == "antialias_area" and self._drag_start_scene_pos is not None:
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
                if self._brush_pan_guard:
                    # Ignore one post-pan pointer sample to avoid accidental long segment jump.
                    self._brush_pan_guard = False
                    event.accept()
                    return
                self._append_brush_point(scene_pos)
            event.accept()
            return
        if self._tool == EditorTool.TRACE_PEN and self._editor_scene.has_pending_polygon():
            last_point = self._editor_scene.pending_last_point()
            target = (
                _snap_to_45(last_point, scene_pos)
                if last_point is not None
                and (
                    self._trace_mode == BrushMode.ANGLED
                    or event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                )
                else scene_pos
            )
            self._editor_scene.update_pending_cursor(target)
            event.accept()
            return
        if self._drag_kind == "vertex" and self._drag_polygon_id is not None and self._drag_vertex_index is not None:
            self._drag_preview_points = self._editor_scene.preview_vertex_move(
                self._drag_polygon_id,
                self._drag_vertex_index,
                scene_pos,
            )
            event.accept()
            return
        if (
            self._drag_kind == "edge"
            and self._drag_polygon_id is not None
            and self._drag_edge_index is not None
            and self._drag_origin_points is not None
            and self._drag_start_scene_pos is not None
        ):
            delta = (
                scene_pos.x() - self._drag_start_scene_pos.x(),
                scene_pos.y() - self._drag_start_scene_pos.y(),
            )
            self._drag_preview_points = self._editor_scene.preview_edge_move(
                self._drag_polygon_id,
                self._drag_edge_index,
                self._drag_origin_points,
                delta,
            )
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
            self._drag_preview_points = self._editor_scene.preview_polygon_move(
                self._drag_polygon_id,
                moved,
            )
            if self._contact_drag_profile is not None:
                self._contact_drag_profile.record_frame(move_event_started_at)
            event.accept()
            return
        super().mouseMoveEvent(event)
