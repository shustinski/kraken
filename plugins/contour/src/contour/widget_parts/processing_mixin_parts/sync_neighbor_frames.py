"""processing_mixin: sync neighbor frames."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
    QTimer,
)


class SyncNeighborFramesMixin:
    def _sync_neighbor_frames(self: Any) -> None:
        if not hasattr(self, "polygon_editor"):
            return
        if (
            self._neighbor_frames_enabled()
            and hasattr(self, "_ensure_neighbor_pyramid_or_prompt")
            and not bool(getattr(self, "_restoring_display_settings", False))
            and bool(self.isVisible())
        ) and not self._ensure_neighbor_pyramid_or_prompt():
            return
        current_path = self._workspace.current_image_path
        session = (
            self._frame_switch_profile_for_path(str(current_path))
            if current_path
            else None
        )
        if session is not None:
            session.mark_pending("neighbor_sync")
        if not self._neighbor_frames_enabled():
            self._neighbor_sync_image_path = None
            self._neighbor_frame_specs = []
            self._neighbor_queued_paths.clear()
            timer = getattr(self, "_neighbor_apply_timer", None)
            if timer is not None:
                timer.stop()
            store = getattr(self, "_pyramid_frame_store", None)
            image_paths = self._ordered_workspace_image_paths()
            frame_id = self._image_path_index(current_path) if current_path else None
            self.polygon_editor.set_pyramid_frame_store(
                store,
                frame_count=len(image_paths),
                columns=max(1, int(self.neighbor_columns_spin.value()))
                if hasattr(self, "neighbor_columns_spin")
                else 1,
                current_frame_id=frame_id,
                enabled=False,
            )
            self.polygon_editor.set_neighbor_frames([], 0.0, 0, False)
            if session is not None:
                session.complete_pending("neighbor_sync", suffix="_disabled")
            QTimer.singleShot(0, self._center_editor_on_current_main_image)
            return
        image_paths = self._ordered_workspace_image_paths()
        mapping = getattr(self, "_image_path_to_index", {})
        normalized_current_path = current_path or ""
        if normalized_current_path and normalized_current_path not in mapping:
            normalized_current_path = str(Path(current_path))
        if not normalized_current_path or normalized_current_path not in mapping:
            self._neighbor_sync_image_path = None
            self._neighbor_frame_specs = []
            self._neighbor_queued_paths.clear()
            timer = getattr(self, "_neighbor_apply_timer", None)
            if timer is not None:
                timer.stop()
            self.polygon_editor.set_neighbor_frames([], 0.0, 0, False)
            if session is not None:
                session.complete_pending("neighbor_sync", suffix="_no_current")
            QTimer.singleShot(0, self._center_editor_on_current_main_image)
            return
        current_index = image_paths.index(normalized_current_path)
        columns = max(1, int(self.neighbor_columns_spin.value()))
        current_row = current_index // columns
        current_column = current_index % columns
        radius = self._neighbor_grid_size() // 2
        previous_sync_path = getattr(self, "_neighbor_sync_image_path", None)
        self._neighbor_sync_image_path = normalized_current_path
        if str(previous_sync_path or "") != normalized_current_path:
            self._neighbor_queued_paths.clear()
            timer = getattr(self, "_neighbor_apply_timer", None)
            if timer is not None:
                timer.stop()
        scene = self.polygon_editor._editor_scene
        scene._pending_neighbor_frames = None
        scene.clear_neighbor_frames()
        specs: list[tuple[int, int, str]] = []
        for row_offset in range(-radius, radius + 1):
            for column_offset in range(-radius, radius + 1):
                if row_offset == 0 and column_offset == 0:
                    continue
                row = current_row + row_offset
                column = current_column + column_offset
                if row < 0 or column < 0 or column >= columns:
                    continue
                index = row * columns + column
                if index < 0 or index >= len(image_paths):
                    continue
                image_path = image_paths[index]
                specs.append((column_offset, row_offset, image_path))
        self._neighbor_frame_specs = specs
        spec_paths = {str(Path(path)) for _column_offset, _row_offset, path in specs}
        self._neighbor_queued_paths.intersection_update(spec_paths)
        self._apply_cached_neighbor_frames()
        for _column_offset, _row_offset, image_path in specs:
            if self._neighbor_frame_image(image_path) is None:
                self._queue_neighbor_frame_load(image_path)
        if getattr(self, "_neighbor_queued_paths", set()):
            self._schedule_neighbor_frame_apply(delay_ms=50)
        elif session is not None:
            session.complete_pending("neighbor_sync", suffix="_cached_only")
