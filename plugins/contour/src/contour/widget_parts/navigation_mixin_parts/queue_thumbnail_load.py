"""navigation_mixin: queue thumbnail load."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    QTimer,
    ThumbnailLoadRunnable,
)


class QueueThumbnailLoadMixin:
    def _queue_thumbnail_load(self: Any, generation: int, path: str) -> None:
        if self._thumbnail_loading_blocked():
            return
        normalized = self._normalized_path(path)
        requested_size = self._thumbnail_request_size()
        if (
            getattr(self, "_thumbnail_loaded_generation", {}).get(normalized) == generation
            and getattr(self, "_thumbnail_loaded_sizes", {}).get(normalized) == requested_size
        ):
            return
        if (
            normalized in self._thumbnail_queued_paths
            and getattr(self, "_thumbnail_queued_sizes", {}).get(normalized) == requested_size
        ):
            return
        self._thumbnail_queued_paths.add(normalized)
        self._thumbnail_queued_sizes[normalized] = requested_size
        runnable = ThumbnailLoadRunnable(
            generation,
            normalized,
            requested_size[0],
            requested_size[1],
            str(getattr(self, "_thumbnail_disk_cache_dir", "")),
        )
        runnable.signals.result.connect(self._on_thumbnail_loaded)
        if not hasattr(self, "_deferred_thumbnail_load_timers"):
            self._deferred_thumbnail_load_timers = []
        timer = QTimer(self)
        timer.setSingleShot(True)

        def _start_thumbnail_load() -> None:
            try:
                if bool(getattr(self, "_closing", False)):
                    self._thumbnail_queued_paths.discard(normalized)
                    self._thumbnail_queued_sizes.pop(normalized, None)
                    return
                if self._thumbnail_loading_blocked():
                    self._thumbnail_queued_paths.discard(normalized)
                    self._thumbnail_queued_sizes.pop(normalized, None)
                    return
                if generation != self._thumbnail_generation:
                    self._thumbnail_queued_paths.discard(normalized)
                    self._thumbnail_queued_sizes.pop(normalized, None)
                    return
                if getattr(self, "_thumbnail_queued_sizes", {}).get(normalized) != requested_size:
                    return
                self._thumbnail_thread_pool.start(runnable)
            finally:
                if hasattr(self, "_deferred_thumbnail_load_timers"):
                    self._deferred_thumbnail_load_timers = [
                        candidate for candidate in self._deferred_thumbnail_load_timers if candidate is not timer
                    ]

        timer.timeout.connect(_start_thumbnail_load)
        self._deferred_thumbnail_load_timers.append(timer)
        timer.start(250)
