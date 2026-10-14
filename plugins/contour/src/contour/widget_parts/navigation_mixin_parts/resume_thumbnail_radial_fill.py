"""navigation_mixin: resume thumbnail radial fill."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    THUMBNAIL_APPLY_INTERVAL_MS,
    THUMBNAIL_MAX_ACTIVE_DECODES,
    THUMBNAIL_RADIAL_LOADS_PER_PUMP,
    THUMBNAIL_RADIAL_PUMP_INTERVAL_MS,
)


class ResumeThumbnailRadialFillMixin:
    def _resume_thumbnail_radial_fill(self: Any) -> None:
        if not self._frame_matrix_thumbnails_enabled():
            return
        if self._thumbnail_loading_blocked():
            return
        # Selection leads the asynchronous workspace load.  Highlight the
        # requested frame immediately, while generation checks below still
        # prevent stale load results from changing the editor.
        current = getattr(self, "_desired_image_path", None) or self._workspace.current_image_path
        if not current:
            return
        normalized = self._normalized_path(current)
        if normalized != getattr(self, "_thumbnail_radial_center_path", None):
            self._reseed_thumbnail_radial_fill()
            return
        paths = getattr(self, "_thumbnail_radial_paths", [])
        cursor = int(getattr(self, "_thumbnail_radial_cursor", 0))
        if cursor < len(paths):
            self._schedule_thumbnail_radial_pump()
            return
        self._reseed_thumbnail_radial_fill()

    def _schedule_thumbnail_radial_pump(self: Any) -> None:
        if not hasattr(self, "_thumbnail_radial_pump_timer"):
            return
        if self._thumbnail_loading_blocked():
            return
        if not getattr(self, "_thumbnail_radial_paths", []):
            return
        active = 0
        try:
            active = int(self._thumbnail_thread_pool.activeThreadCount())
        except AttributeError:
            pass
        active += len(getattr(self, "_thumbnail_queued_paths", set()))
        active += len(getattr(self, "_thumbnail_pending_apply", {}))
        if active >= int(THUMBNAIL_MAX_ACTIVE_DECODES):
            self._thumbnail_radial_pump_timer.stop()
            self._thumbnail_radial_pump_timer.start(max(80, int(THUMBNAIL_RADIAL_PUMP_INTERVAL_MS)))
            return
        self._thumbnail_radial_pump_timer.stop()
        self._thumbnail_radial_pump_timer.start(max(16, int(THUMBNAIL_RADIAL_PUMP_INTERVAL_MS)))

    def _pump_thumbnail_radial_loads(self: Any) -> None:
        if self._thumbnail_loading_blocked():
            return
        paths: list[str] = getattr(self, "_thumbnail_radial_paths", [])
        cursor = int(getattr(self, "_thumbnail_radial_cursor", 0))
        if cursor >= len(paths):
            return
        generation = self._thumbnail_generation
        per_pump = max(1, int(THUMBNAIL_RADIAL_LOADS_PER_PUMP))
        queued = 0
        while cursor < len(paths) and queued < per_pump:
            path = paths[cursor]
            cursor += 1
            before = len(self._thumbnail_queued_paths)
            self._queue_thumbnail_load(generation, path)
            if len(self._thumbnail_queued_paths) > before:
                queued += 1
        self._thumbnail_radial_cursor = cursor
        if cursor < len(paths):
            self._schedule_thumbnail_radial_pump()

    def _schedule_thumbnail_icon_apply(self: Any) -> None:
        if not hasattr(self, "_thumbnail_apply_timer"):
            return
        self._thumbnail_apply_timer.stop()
        self._thumbnail_apply_timer.start(max(16, int(THUMBNAIL_APPLY_INTERVAL_MS)))
