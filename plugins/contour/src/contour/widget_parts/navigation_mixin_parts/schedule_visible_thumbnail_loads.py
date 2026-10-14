"""navigation_mixin: schedule visible thumbnail loads."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    THUMBNAIL_MAX_ACTIVE_DECODES,
    QSize,
    Qt,
    QTimer,
)


class ScheduleVisibleThumbnailLoadsMixin:
    def _thumbnail_request_size(self: Any) -> tuple[int, int]:
        if hasattr(self, "thumbnail_grid") and hasattr(self.thumbnail_grid, "thumbnailSourceSize"):
            size = self.thumbnail_grid.thumbnailSourceSize()
            return (max(1, int(size.width())), max(1, int(size.height())))
        icon_size = self._thumbnail_icon_size if hasattr(self, "_thumbnail_icon_size") else QSize(64, 48)
        from ...ui.large_dataset import clamp_thumbnail_source_size

        return clamp_thumbnail_source_size(int(icon_size.width()), int(icon_size.height()))

    def _visible_thumbnail_rows(self: Any, *, buffer_rows: int = 2) -> tuple[int, int]:
        if not hasattr(self, "thumbnail_grid_scroll_area"):
            return (0, -1)
        icon_size = self._thumbnail_icon_size if hasattr(self, "_thumbnail_icon_size") else QSize(64, 48)
        row_step = max(1, int(icon_size.height()))
        if hasattr(self.thumbnail_grid, "_cell_step_y"):
            row_step = max(1, int(self.thumbnail_grid._cell_step_y()))
        vertical = self.thumbnail_grid_scroll_area.verticalScrollBar()
        viewport_h = max(1, int(self.thumbnail_grid_scroll_area.viewport().height()))
        scroll_value = int(vertical.value())
        first_row = max(0, (scroll_value // row_step) - buffer_rows)
        last_row = max(first_row, ((scroll_value + viewport_h) // row_step) + buffer_rows)
        return first_row, last_row

    def _visible_thumbnail_indexes(self: Any) -> list[int]:
        if not self._frame_matrix_thumbnails_enabled() or not hasattr(self, "thumbnail_grid"):
            return []
        columns = max(1, self._thumbnail_columns())
        first_row, last_row = self._visible_thumbnail_rows()
        count = self.thumbnail_grid.count()
        indexes: set[int] = set()
        for row in range(first_row, last_row + 1):
            start = row * columns
            end = min(count, start + columns)
            indexes.update(range(start, end))
        current = self._workspace.current_image_path
        if current:
            center = getattr(self, "_thumbnail_path_to_row", {}).get(self._normalized_path(current))
            if center is not None:
                current_row = center // columns
                for row in range(max(0, current_row - 2), current_row + 3):
                    start = row * columns
                    end = min(count, start + columns)
                    indexes.update(range(start, end))
        return sorted(index for index in indexes if 0 <= index < count)

    def _schedule_visible_thumbnail_loads(self: Any) -> None:
        if self._thumbnail_loading_blocked() or not hasattr(self, "thumbnail_grid"):
            return
        generation = self._thumbnail_generation
        queued_paths: list[str] = []
        for index in self._visible_thumbnail_indexes():
            item = self.thumbnail_grid.item(index)
            if item is None or item.isHidden():
                continue
            path = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if path:
                queued_paths.append(self._normalized_path(path))
        keep = set(queued_paths)
        current = self._workspace.current_image_path
        if current:
            keep.add(self._normalized_path(current))
        icon_cache = getattr(self, "_thumbnail_icon_cache", {})
        cache_limit = max(1, int(getattr(self, "_thumbnail_icon_cache_limit", 2048)))
        if len(icon_cache) > cache_limit:
            for key in list(icon_cache.keys()):
                key_path = key[0] if isinstance(key, tuple) and key else key
                if key_path not in keep:
                    icon_cache.pop(key, None)
                if len(icon_cache) <= cache_limit:
                    break
        decode_budget = max(0, int(THUMBNAIL_MAX_ACTIVE_DECODES))
        try:
            decode_budget -= int(self._thumbnail_thread_pool.activeThreadCount())
        except AttributeError:
            pass
        decode_budget -= len(getattr(self, "_thumbnail_queued_paths", set()))
        decode_budget -= len(getattr(self, "_thumbnail_pending_apply", {}))
        deferred = False
        for path in queued_paths:
            if decode_budget <= 0:
                deferred = True
                break
            before = len(self._thumbnail_queued_paths)
            self._queue_thumbnail_load(generation, path)
            if len(self._thumbnail_queued_paths) > before:
                decode_budget -= 1
        if deferred:
            self._schedule_visible_thumbnail_loads_debounced()

    def _on_thumbnail_loaded(self: Any, generation: int, path: str, width: int, height: int, qimage: object) -> None:
        normalized_path = self._normalized_path(path)
        self._thumbnail_queued_paths.discard(normalized_path)
        self._thumbnail_queued_sizes.pop(normalized_path, None)
        if generation != self._thumbnail_generation or not hasattr(self, "thumbnail_grid"):
            return
        if not self._frame_matrix_thumbnails_enabled():
            return
        requested_size = (max(1, int(width)), max(1, int(height)))
        pending = getattr(self, "_thumbnail_pending_apply", {})
        pending[normalized_path] = (qimage, requested_size, generation)
        icon_cache = getattr(self, "_thumbnail_icon_cache", {})
        icon_cache.pop(normalized_path, None)
        icon_cache.pop((normalized_path, requested_size), None)
        self._schedule_thumbnail_icon_apply()

    def _prime_visible_thumbnail_reload(self: Any) -> None:
        if not hasattr(self, "thumbnail_grid"):
            return
        loaded_generation = getattr(self, "_thumbnail_loaded_generation", {})
        loaded_sizes = getattr(self, "_thumbnail_loaded_sizes", {})
        icon_cache = getattr(self, "_thumbnail_icon_cache", {})
        paths: set[str] = set()
        for index in self._visible_thumbnail_indexes():
            item = self.thumbnail_grid.item(index)
            if item is None or item.isHidden():
                continue
            path = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if path:
                paths.add(self._normalized_path(path))
        current = self._workspace.current_image_path
        if current:
            paths.add(self._normalized_path(current))
        if not paths:
            return
        for path in paths:
            loaded_generation.pop(path, None)
            loaded_sizes.pop(path, None)
        for key in list(icon_cache):
            if key in paths or (isinstance(key, tuple) and key and key[0] in paths):
                icon_cache.pop(key, None)

    def _resume_frame_matrix_thumbnail_loading(self: Any, *, radial_fill: bool = True) -> None:
        if not self._frame_matrix_thumbnails_enabled() or not hasattr(self, "thumbnail_grid"):
            return
        if getattr(self, "_frame_switch_profile", None) is not None:
            return
        if self._thumbnail_loading_blocked():
            return
        self._schedule_thumbnail_icon_apply()
        QTimer.singleShot(0, self._schedule_visible_thumbnail_loads)
        if radial_fill:
            QTimer.singleShot(0, self._reseed_thumbnail_radial_fill)
