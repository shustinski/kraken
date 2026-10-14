"""navigation_mixin: rebuild thumbnail grid."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    QTimer,
)


class RebuildThumbnailGridMixin:
    def _rebuild_thumbnail_grid(self: Any) -> None:
        if not self._frame_matrix_enabled():
            self._disable_frame_matrix_runtime()
            return
        if not hasattr(self, "thumbnail_grid"):
            return
        self._cancel_thumbnail_loading()
        self._thumbnail_grid_build_generation = int(
            getattr(self, "_thumbnail_grid_build_generation", 0)
        ) + 1
        build_generation = self._thumbnail_grid_build_generation
        self._thumbnail_loaded_generation.clear()
        self._thumbnail_rebuild_in_progress = True
        self._thumbnail_selected_path = None
        self._thumbnail_path_to_row: dict[str, int] = {}
        try:
            self._thumbnail_thread_pool.clear()
        except AttributeError:
            pass
        self.thumbnail_grid.blockSignals(True)
        self.thumbnail_grid._suppress_matrix_refresh = True
        self.thumbnail_grid.setUpdatesEnabled(False)
        try:
            self.thumbnail_grid.clear()
        finally:
            self.thumbnail_grid.blockSignals(False)
        paths = self._thumbnail_paths_for_matrix()
        configured_chunk_size = max(1, int(getattr(self, "_thumbnail_build_chunk_size", 50)))
        max_ui_chunk_size = 250 if self._uses_large_frame_list() else 100
        chunk_size = min(configured_chunk_size, max_ui_chunk_size)
        default_interval = 25
        if self._uses_large_frame_list():
            default_interval = 8
        chunk_interval_ms = max(0, int(getattr(self, "_thumbnail_build_interval_ms", default_interval)))
        if not paths:
            QTimer.singleShot(0, self._finish_thumbnail_grid_build)
            return

        def _add_thumbnail_chunk(start: int) -> None:
            if (
                build_generation != getattr(self, "_thumbnail_grid_build_generation", 0)
                or bool(getattr(self, "_closing", False))
                or not hasattr(self, "thumbnail_grid")
            ):
                return
            end = min(len(paths), start + chunk_size)
            self.thumbnail_grid.blockSignals(True)
            try:
                for path in paths[start:end]:
                    row = self.thumbnail_grid.count()
                    self.thumbnail_grid.addItem(self._make_thumbnail_grid_item(path))
                    self._thumbnail_path_to_row[str(path)] = row
            finally:
                self.thumbnail_grid.blockSignals(False)
            if end < len(paths):
                QTimer.singleShot(chunk_interval_ms, lambda next_start=end: _add_thumbnail_chunk(next_start))
                return
            self._finish_thumbnail_grid_build()

        QTimer.singleShot(0, lambda: _add_thumbnail_chunk(0))
