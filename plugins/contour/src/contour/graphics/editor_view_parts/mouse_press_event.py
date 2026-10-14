"""editor_view: mouse press event."""
from __future__ import annotations
from PyQt6.QtCore import (
    QPointF,
    Qt,
)
from PyQt6.QtGui import (
    QMouseEvent,
)
from contour.graphics.tool_mode_logic import (
    is_via_polygon,
)
from contour.graphics.tools import DeleteVertexMode, EditorTool, PolygonCreateMode


class MousePressEventMixin:
    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ):
            self.setFocus(Qt.FocusReason.MouseFocusReason)
        viewport_pixel = self._require_viewport().mapFrom(self, event.position().toPoint())
        self._last_pointer_viewport_pos = QPointF(viewport_pixel)
        scene_pos = self.mapToScene(self._viewport_to_view_point(viewport_pixel))
        self._last_pointer_scene_pos = scene_pos
        tolerance = self._scene_tolerance(8)

        if event.button() == Qt.MouseButton.MiddleButton:
            if self._drag_kind == "brush":
                self._brush_pan_guard = True
            self._middle_pan_active = True
            self._middle_pan_last_viewport = QPointF(viewport_pixel)
            self._set_polygon_overlay_hide_hold(
                "middle",
                True,
                hide_gradient_overlay=False,
            )
            event.accept()
            return

        if self._paste_mode and event.button() == Qt.MouseButton.LeftButton:
            if self.vector_edits_locked():
                self._exit_paste_mode()
                event.accept()
                return
            pasted_ids = self._editor_scene.add_cloned_polygons_at(
                self._clipboard_polygons,
                self._clipboard_anchor,
                scene_pos,
            )
            self._update_paste_preview(scene_pos)
            self.contactPasteFinished.emit(len(pasted_ids))
            event.accept()
            return

        ctrl_region_selection = self._ctrl_image_region_selection_enabled and bool(
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        )
        if (self._image_region_selection_mode or ctrl_region_selection) and event.button() == Qt.MouseButton.LeftButton:
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
            if create_mode == PolygonCreateMode.RECTANGLE and event.button() in (
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.RightButton,
            ):
                if self._drag_kind == "rect_polygon":
                    started_with_right = self._drag_erases
                    clicked_right = event.button() == Qt.MouseButton.RightButton
                    if clicked_right == started_with_right:
                        self._commit_rectangle_polygon(scene_pos)
                    else:
                        self._cancel_rectangle_polygon()
                    event.accept()
                    return
                self._start_rectangle_polygon(
                    scene_pos,
                    erase=event.button() == Qt.MouseButton.RightButton,
                )
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
            self._start_brush_drag(scene_pos, erase=event.button() == Qt.MouseButton.RightButton)
            event.accept()
            return

        if self._tool == EditorTool.TRACE_PEN and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        ):
            self._append_trace_point(
                scene_pos,
                erase=event.button() == Qt.MouseButton.RightButton,
                snap=bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier),
            )
            event.accept()
            return

        if self._tool == EditorTool.ADD_VIA and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        ):
            if event.button() == Qt.MouseButton.LeftButton:
                self.contactPlacementAttemptStarted.emit()
            polygon_id = self._editor_scene.polygon_at(scene_pos)
            polygon = self._editor_scene.polygon_snapshot(polygon_id) if polygon_id is not None else None
            if event.button() == Qt.MouseButton.RightButton:
                self.contactDeletionStarted.emit(1)
                deleted = self._editor_scene.delete_via_at(scene_pos)
                if deleted:
                    self._emit_recognized_vias_deleted([polygon] if polygon is not None else [])
                self.contactDeletionFinished.emit(1 if deleted else 0)
                self.contactPlacementAttemptFinished.emit(False)
            elif (
                self._contact_recognition_mode
                and polygon is not None
                and is_via_polygon(polygon)
            ):
                self._editor_scene.select_polygon(polygon.id)
                if self._via_debug_inspection_enabled:
                    self.viaDebugRequested.emit(polygon)
                self.contactPlacementAttemptFinished.emit(False)
            else:
                added = self._editor_scene.add_via_at(scene_pos, self._via_width, self._via_height)
                if added:
                    self.manualViaAdded.emit(float(scene_pos.x()), float(scene_pos.y()))
                self.contactPlacementAttemptFinished.emit(added)
            self._update_tool_cursors()
            event.accept()
            return

            self._update_tool_cursors()
            event.accept()
            return

        if event.button() == Qt.MouseButton.RightButton:
            polygon_id = self._editor_scene.polygon_at(scene_pos)
            polygon = self._editor_scene.polygon_snapshot(polygon_id)
            is_contact = polygon is not None and is_via_polygon(polygon)
            if is_contact:
                self.contactDeletionStarted.emit(1)
            deleted = self._editor_scene.delete_via_at(scene_pos)
            if deleted:
                self._emit_recognized_vias_deleted([polygon] if polygon is not None else [])
                event.accept()
            if is_contact:
                self.contactDeletionFinished.emit(1 if deleted else 0)
                return
            super().mousePressEvent(event)
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
            polygon_id = self._editor_scene.polygon_at(scene_pos)
            polygon = self._editor_scene.polygon_snapshot(polygon_id)
            is_contact = polygon is not None and is_via_polygon(polygon)
            if is_contact:
                self.contactDeletionStarted.emit(1)
            deleted = self._editor_scene.delete_polygon_at(scene_pos)
            if is_contact:
                self.contactDeletionFinished.emit(1 if deleted else 0)
            event.accept()
            return

        if self._tool == EditorTool.ANTIALIAS:
            self._drag_kind = "antialias_area"
            self._drag_start_scene_pos = scene_pos
            self._editor_scene.set_preview_rect(scene_pos, scene_pos)
            event.accept()
            return

        if self._tool == EditorTool.SELECT:
            additive_selection = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            polygon_id = self._editor_scene.polygon_at(scene_pos, cycle=not additive_selection)
            if polygon_id is None:
                neighbor_path = self._editor_scene.neighbor_frame_path_at(scene_pos)
                if neighbor_path is not None:
                    event.accept()
                    return
                target_frame_id = self._pyramid_frame_at_viewport_pos(viewport_pixel)
                if target_frame_id is not None and target_frame_id != self._pyramid_current_frame_id:
                    if not self._confirm_frame_navigation():
                        event.accept()
                        return
                    self.set_current_frame_id(target_frame_id, center=False, emit_signal=True)
                    self.frameNavigationRequested.emit(target_frame_id)
                    event.accept()
                    return
                self._drag_kind = "select_area"
                self.contactMultiSelectionStarted.emit()
                self._drag_start_scene_pos = scene_pos
                self._select_press_polygon_id = None
                self._select_press_start = None
                self._editor_scene.set_preview_rect(scene_pos, scene_pos)
                event.accept()
                return
            if additive_selection:
                self.contactMultiSelectionStarted.emit()
                self.contactMultiSelectionApplyStarted.emit()
            self._editor_scene.select_polygon(polygon_id, additive=additive_selection)
            if additive_selection:
                self.contactMultiSelectionFinished.emit(self._selected_contact_count())
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
            clicked_polygon_id = self._editor_scene.polygon_at_nearest_edge(scene_pos, tolerance)
            selected_polygon_id = self._editor_scene.selected_polygon_id()
            if selected_polygon_id is None:
                if clicked_polygon_id is not None:
                    self._editor_scene.select_polygon(clicked_polygon_id)
                event.accept()
                return
            if clicked_polygon_id is not None and clicked_polygon_id != selected_polygon_id:
                self._editor_scene.select_polygon(clicked_polygon_id)
                event.accept()
                return
            self._editor_scene.add_vertex_at(selected_polygon_id, scene_pos)
            event.accept()
            return

        if self._tool == EditorTool.DELETE_VERTEX:
            if self._delete_vertex_mode == DeleteVertexMode.AREA:
                self._drag_kind = "delete_area"
                self._drag_start_scene_pos = scene_pos
                self._editor_scene.preview_delete_vertices_in_rect(scene_pos, scene_pos)
                event.accept()
                return
            self._editor_scene.delete_vertex_at(scene_pos, tolerance)
            event.accept()
            return

        if self._tool == EditorTool.MOVE_VERTEX:
            selected_polygon_id = self._editor_scene.selected_polygon_id()
            clicked_polygon_id = self._editor_scene.polygon_at(scene_pos)
            if clicked_polygon_id is not None and clicked_polygon_id != selected_polygon_id:
                self._editor_scene.select_polygon(clicked_polygon_id)
            target = self._editor_scene.pick_move_target(
                scene_pos,
                self._scene_tolerance(8),
                self._scene_tolerance(10),
            )
            if target is not None:
                kind, polygon_id, target_index = target
                self._editor_scene.select_polygon(polygon_id)
                self._editor_scene.clear_move_target_preview()
                self._drag_polygon_id = polygon_id
                self._drag_origin_points = self._editor_scene.polygon_points(polygon_id)
                self._drag_preview_points = list(self._drag_origin_points)
                self._editor_scene.begin_polygon_edit_preview(polygon_id)
                self._drag_polygons_snapshot = self._editor_scene.get_polygons()
                self._drag_start_scene_pos = scene_pos
                if kind == "vertex":
                    self._drag_kind = "vertex"
                    self._drag_vertex_index = target_index
                    self._drag_edge_index = None
                else:
                    self._drag_kind = "edge"
                    self._drag_edge_index = target_index
                    self._drag_vertex_index = None
            event.accept()
            return
