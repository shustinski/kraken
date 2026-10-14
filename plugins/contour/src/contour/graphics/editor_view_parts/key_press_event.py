"""editor_view: key press event."""
from __future__ import annotations
from PyQt6.QtGui import (
    QKeyEvent,
)
from contour.graphics.editor_hotkeys import event_matches_shortcut
from contour.graphics.tool_mode_logic import (
    is_via_polygon,
)
from contour.graphics.tools import EditorTool


class KeyPressEventMixin:
    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is None:
            return
        if (
            event_matches_shortcut(event, "fit_view", "F")
            and self.isEnabled()
        ):
            if not event.isAutoRepeat():
                self.fit_to_view()
            event.accept()
            return
        if (
            event_matches_shortcut(event, "hold_vectors", "Space")
            and self.isEnabled()
            and (self.hasFocus() or self._require_viewport().hasFocus())
        ):
            if event.isAutoRepeat():
                event.accept()
                return
            if "space" not in self._polygon_overlay_hide_holds:
                self._set_polygon_overlay_hide_hold("space", True)
            event.accept()
            return
        if (
            event_matches_shortcut(event, "hold_source_image", "X")
            and self.isEnabled()
            and (self.hasFocus() or self._require_viewport().hasFocus())
        ):
            if event.isAutoRepeat():
                event.accept()
                return
            if not self._filter_preview_hold_active:
                self._filter_preview_hold_active = True
                self.filterPreviewHoldChanged.emit(True)
            event.accept()
            return
        if (
            event_matches_shortcut(event, "finish_polygon", "Enter")
            and self._tool == EditorTool.ADD_VERTEX
        ):
            self._editor_scene.select_polygon(None)
            event.accept()
            return
        if (
            event_matches_shortcut(event, "finish_polygon", "Enter")
            and self._tool in (EditorTool.ADD_POLYGON, EditorTool.TRACE_PEN)
            and self._editor_scene.has_pending_polygon()
        ):
            if self._tool == EditorTool.TRACE_PEN:
                self._finish_pending_trace()
            else:
                self._finish_pending_polygon()
            event.accept()
            return
        if event_matches_shortcut(event, "cancel", "Esc"):
            paste_was_active = self._paste_mode
            if self._drag_kind == "select_area":
                self.contactMultiSelectionFinished.emit(0)
            if self._contact_drag_profile is not None:
                self._finish_contact_drag_profile("cancelled", commit_ms=0.0)
            self._editor_scene.cancel_pending_polygon()
            self._editor_scene.clear_measurement()
            self._editor_scene.clear_preview_rect()
            self._exit_paste_mode()
            if paste_was_active:
                self.contactPasteFinished.emit(0)
            if self._tool in (EditorTool.SELECT, EditorTool.ADD_VERTEX):
                self._editor_scene.select_polygon(None)
            self._select_press_polygon_id = None
            self._select_press_start = None
            if self._tool == EditorTool.RULER:
                self.rulerMeasurementChanged.emit("")
            self._drag_kind = None
            self._drag_erases = False
            self._drag_polygon_is_contact = False
            self._pending_polygon_erases = None
            self._update_tool_cursors()
            self._emit_effective_polygon_create_mode_changed()
            event.accept()
            return
        if event_matches_shortcut(event, "delete_selection", "Del"):
            if self.vector_edits_locked():
                event.accept()
                return
            deleted = self._editor_scene.selected_deletable_polygons()
            contacts = [polygon for polygon in deleted if is_via_polygon(polygon)]
            if contacts:
                self.contactDeletionStarted.emit(len(contacts))
            did_delete = self._editor_scene.delete_polygon()
            if did_delete:
                self._emit_recognized_vias_deleted(deleted)
            if contacts:
                self.contactDeletionFinished.emit(len(contacts) if did_delete else 0)
            event.accept()
            return
        if (
            event_matches_shortcut(event, "cycle_tool_mode", "Shift")
            and self._drag_kind is None
            and not event.isAutoRepeat()
            and self._cycle_active_tool_mode()
        ):
            event.accept()
            return
        if (
            event_matches_shortcut(event, "cycle_tool_mode", "Shift")
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
