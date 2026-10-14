"""editor_view: key release event."""
from __future__ import annotations
from time import perf_counter
from PyQt6.QtCore import (
    QPoint,
)
from PyQt6.QtGui import (
    QKeyEvent,
    QWheelEvent,
)
from contour.domain import PolygonData
from ...infrastructure.contact_placement_profiler import (
    SceneZoomProfile,
)
from ...infrastructure.profiling import (
    scene_zoom_profiling_enabled,
    write_profile_report,
)
from contour.graphics.editor_hotkeys import event_matches_shortcut
from contour.graphics.tool_mode_logic import (
    is_via_polygon,
)
from contour.graphics.tools import EditorTool
from contour.graphics.viewport_navigation import (
    clamp_zoom_factor,
)

from contour.graphics.editor_view_parts.module_state import (
    _ZOOM_EASING_FRACTION,
    _ZOOM_SETTLE_RATIO,
)


class KeyReleaseEventMixin:
    def _emit_recognized_vias_deleted(self, polygons: list[PolygonData]) -> None:
        recognized = [
            polygon.clone()
            for polygon in polygons
            if is_via_polygon(polygon) and polygon.recognition_score is not None
        ]
        if recognized:
            self.recognizedViasDeleted.emit(recognized)

    def _selected_contact_count(self) -> int:
        return self._editor_scene.selected_contact_count()

    def selected_object_counts(self) -> tuple[int, int]:
        return self._editor_scene.selected_object_counts()

    def keyReleaseEvent(self, event: QKeyEvent | None) -> None:
        if event is None:
            return
        if (
            event_matches_shortcut(event, "hold_vectors", "Space")
            and not event.isAutoRepeat()
            and "space" in self._polygon_overlay_hide_holds
        ):
            self._set_polygon_overlay_hide_hold("space", False)
            event.accept()
            return
        if (
            event_matches_shortcut(event, "hold_source_image", "X")
            and not event.isAutoRepeat()
            and self._filter_preview_hold_active
        ):
            self._filter_preview_hold_active = False
            self.filterPreviewHoldChanged.emit(False)
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
        super().keyReleaseEvent(event)

    def _wheel_event_viewport_pixel(self, event: QWheelEvent) -> QPoint:
        # QGraphicsView::viewportEvent forwards wheel events with viewport-local position.
        return event.position().toPoint()

    def _viewport_to_view_point(self, viewport_pixel: QPoint) -> QPoint:
        return self._require_viewport().mapTo(self, viewport_pixel)

    def _view_to_viewport_point(self, view_pixel: QPoint) -> QPoint:
        return self._require_viewport().mapFrom(self, view_pixel)

    def _zoom_focus_viewport_pixel(self) -> QPoint:
        if self._last_pointer_viewport_pos is not None:
            p = self._last_pointer_viewport_pos.toPoint()
            vr = self._require_viewport().rect()
            clamped = QPoint(p.x(), p.y())
            if not vr.contains(clamped):
                return vr.center()
            return clamped
        return self._require_viewport().rect().center()

    def _queue_wheel_zoom(self, viewport_pixel: QPoint, factor: float) -> None:
        if factor == 1.0 or factor <= 0:
            return
        self._pending_wheel_zoom_factor *= factor
        self._pending_wheel_zoom_viewport_pixel = QPoint(viewport_pixel)
        if self._wheel_zoom_timer.isActive():
            return
        self._wheel_zoom_timer.start()

    def _flush_queued_wheel_zoom(self) -> None:
        factor = self._pending_wheel_zoom_factor
        viewport_pixel = self._pending_wheel_zoom_viewport_pixel
        self._pending_wheel_zoom_factor = 1.0
        self._pending_wheel_zoom_viewport_pixel = None
        if viewport_pixel is None or factor == 1.0 or factor <= 0:
            return
        self._start_zoom_animation(viewport_pixel, factor)

    def _start_zoom_animation(self, viewport_pixel: QPoint, factor: float) -> None:
        if factor == 1.0 or factor <= 0:
            return
        current_zoom = self.zoom_factor()
        base_zoom = self._zoom_animation_target_zoom if self._zoom_animation_timer.isActive() else current_zoom
        target_zoom = clamp_zoom_factor(base_zoom * float(factor))
        if abs(target_zoom - current_zoom) <= 1e-9:
            return
        self._editor_scene.pause_deferred_geometry_repair()
        self._start_scene_zoom_profile(current_zoom, target_zoom)
        self._zoom_animation_viewport_pixel = QPoint(viewport_pixel)
        self._zoom_animation_target_zoom = target_zoom
        self._update_navigation_scene_rect(target_zoom)
        self._enter_zoom_render_mode()
        if not self._zoom_animation_timer.isActive():
            self._zoom_animation_timer.start()

    def _stop_zoom_animation(self) -> None:
        self._zoom_animation_timer.stop()
        self._zoom_animation_viewport_pixel = None
        self._leave_zoom_render_mode()
        self._finish_scene_zoom_profile("interrupted")
        self._editor_scene.resume_deferred_geometry_repair()

    def _advance_zoom_animation(self) -> None:
        frame_started_at = perf_counter()
        viewport_pixel = self._zoom_animation_viewport_pixel
        if viewport_pixel is None:
            self._finish_zoom_animation(frame_started_at=frame_started_at)
            return
        current_zoom = self.zoom_factor()
        target_zoom = self._zoom_animation_target_zoom
        remaining = target_zoom - current_zoom
        if abs(remaining) <= max(_ZOOM_SETTLE_RATIO, abs(target_zoom) * _ZOOM_SETTLE_RATIO):
            next_zoom = target_zoom
            finish = True
        else:
            next_zoom = current_zoom + remaining * _ZOOM_EASING_FRACTION
            finish = False
        factor = next_zoom / current_zoom if current_zoom > 0 else 1.0
        self._apply_zoom_at_viewport_pixel(viewport_pixel, factor, update_navigation=False)
        self.zoomChanged.emit(self.zoom_factor())
        if finish:
            self._finish_zoom_animation(frame_started_at=frame_started_at)
        else:
            self._require_viewport().update()
            if self._scene_zoom_profile is not None:
                self._scene_zoom_profile.record_frame(frame_started_at)

    def _finish_zoom_animation(self, *, frame_started_at: float | None = None) -> None:
        self._zoom_animation_timer.stop()
        self._zoom_animation_viewport_pixel = None
        self._leave_zoom_render_mode()
        self._update_navigation_scene_rect()
        self._schedule_pyramid_visible_update()
        self._refresh_gradient_field_arrows()
        self._update_tool_cursors()
        self._require_viewport().update()
        if frame_started_at is not None and self._scene_zoom_profile is not None:
            self._scene_zoom_profile.record_frame(frame_started_at)
        self._finish_scene_zoom_profile("displayed")
        self._editor_scene.resume_deferred_geometry_repair()
        if (
            self._tool == EditorTool.MOVE_VERTEX
            and self._drag_kind is None
            and self._last_pointer_scene_pos is not None
        ):
            self._editor_scene.sync_move_target_preview(
                self._last_pointer_scene_pos,
                vertex_tolerance=self._scene_tolerance(8),
                edge_tolerance=self._scene_tolerance(10),
            )

    def _start_scene_zoom_profile(
        self,
        initial_zoom: float,
        target_zoom: float,
    ) -> None:
        profile = self._scene_zoom_profile
        if profile is not None:
            profile.update_target(target_zoom)
            return
        if not scene_zoom_profiling_enabled():
            return
        self._scene_zoom_profile = SceneZoomProfile.begin(
            initial_zoom=initial_zoom,
            target_zoom=target_zoom,
        )
        write_profile_report(
            "[contour scene zoom profiling] "
            f"started zoom={initial_zoom:.4f} target={target_zoom:.4f}",
        )

    def _finish_scene_zoom_profile(self, status: str) -> None:
        profile = self._scene_zoom_profile
        if profile is None:
            return
        self._scene_zoom_profile = None
        profile.finish()
        write_profile_report(
            profile.format_summary(status=status, final_zoom=self.zoom_factor()),
            profile.format_stats(),
        )
