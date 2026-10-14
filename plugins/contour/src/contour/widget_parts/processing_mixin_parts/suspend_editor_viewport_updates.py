"""processing_mixin: suspend editor viewport updates."""
from __future__ import annotations
from typing import Any
from ...adapters.qt.object_validity import qt_object_is_valid
from contour.widget_parts._imports import (
    FrameLoadPayload,
    Path,
    QTimer,
    VectorGeometrySettings,
    polygons_needing_repair,
)


class SuspendEditorViewportUpdatesMixin:
    def _make_polygon_repair_scanner(self: Any, *, include_ring_reasons: bool = False):
        settings = (
            self._vector_geometry_settings_from_widgets()
            if hasattr(self, "_vector_geometry_settings_from_widgets")
            else VectorGeometrySettings()
        )

        def _scan(polygons: list) -> dict[int, list[str]]:
            return polygons_needing_repair(
                list(polygons),
                settings,
                include_ring_reasons=include_ring_reasons,
            )

        return _scan

    def _payload_repair_reasons(self: Any, payload: FrameLoadPayload) -> dict[int, list[str]] | None:
        reasons = getattr(payload, "repair_reasons", None)
        if reasons is None:
            return None
        return {int(polygon_id): list(codes) for polygon_id, codes in reasons.items()}

    def _editor_polygons_are_current_frame(self: Any) -> bool:
        current_path = self._workspace.current_image_path
        if not current_path:
            return False
        if getattr(self, "_pending_editor_frame_apply", None) is not None:
            return False
        signature = getattr(self, "_editor_polygons_signature", None)
        if not isinstance(signature, tuple) or not signature:
            return False
        return str(Path(signature[0])) == str(Path(current_path))

    def _sync_editor_polygons_to_current_workspace(self: Any) -> bool:
        if not self._editor_polygons_are_current_frame():
            return False
        return self._workspace.update_current_polygons(self.get_polygons())

    def _neighbor_frames_enabled(self: Any) -> bool:
        return bool(
            hasattr(self, "show_neighbor_frames_checkbox")
            and self.show_neighbor_frames_checkbox.isChecked()
        )

    def _neighbor_vectors_enabled(self: Any) -> bool:
        return bool(
            self._neighbor_frames_enabled()
            and hasattr(self, "show_neighbor_vectors_checkbox")
            and self.show_neighbor_vectors_checkbox.isChecked()
        )

    def _request_neighbor_frame_sync(self: Any, *, delay_ms: int = 0) -> None:
        if not self._neighbor_frames_enabled():
            timer = getattr(self, "_neighbor_sync_timer", None)
            if timer is not None:
                timer.stop()
            self._sync_neighbor_frames()
            return
        timer = getattr(self, "_neighbor_sync_timer", None)
        if timer is None:
            QTimer.singleShot(max(0, int(delay_ms)), self._sync_neighbor_frames)
            return
        timer.stop()
        timer.start(max(0, int(delay_ms)))

    def _schedule_neighbor_frames_after_main_image_ready(self: Any) -> None:
        self._request_neighbor_frame_sync(delay_ms=0)

    def _clear_neighbor_frame_display_for_frame_change(self: Any) -> None:
        self._neighbor_sync_image_path = None
        self._neighbor_frame_specs: list[tuple[int, int, str]] = []
        self._neighbor_queued_paths.clear()
        timer = getattr(self, "_neighbor_apply_timer", None)
        if timer is not None:
            timer.stop()
        if not hasattr(self, "polygon_editor"):
            return
        scene = getattr(self.polygon_editor, "_editor_scene", None)
        if scene is None:
            return
        scene._pending_neighbor_frames = None
        scene.clear_neighbor_frames()

    def _neighbor_sync_is_current(self: Any) -> bool:
        sync_path = getattr(self, "_neighbor_sync_image_path", None)
        current_path = self._workspace.current_image_path
        if not sync_path or not current_path:
            return False
        return str(Path(sync_path)) == str(Path(current_path))

    def _suspend_editor_viewport_updates(self: Any, callback) -> None:
        """Apply editor mutations without intermediate paints (image+vectors swap)."""

        editor = getattr(self, "polygon_editor", None)
        if editor is None or not qt_object_is_valid(editor):
            callback()
            return
        editor.setUpdatesEnabled(False)
        try:
            callback()
        finally:
            if qt_object_is_valid(editor):
                editor.setUpdatesEnabled(True)
                viewport = editor.viewport()
                if viewport is not None:
                    viewport.update()
