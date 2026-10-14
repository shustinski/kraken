"""navigation_mixin: reseed thumbnail radial fill."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
    QListWidgetItem,
    Qt,
    QTimer,
)


class ReseedThumbnailRadialFillMixin:
    def _finish_thumbnail_grid_build(self: Any) -> None:
        self._thumbnail_rebuild_in_progress = False
        self._configure_thumbnail_grid_geometry()
        self._rebuild_thumbnail_path_index()
        self._apply_asset_view_filter()
        self._update_thumbnail_grid_selection()
        self.thumbnail_grid._suppress_matrix_refresh = False
        self.thumbnail_grid.setUpdatesEnabled(True)
        self._flush_thumbnail_icon_batch()
        self._schedule_visible_thumbnail_loads_debounced()

    def _rebuild_thumbnail_path_index(self: Any) -> None:
        mapping: dict[str, int] = {}
        if hasattr(self, "thumbnail_grid"):
            for index in range(self.thumbnail_grid.count()):
                item = self.thumbnail_grid.item(index)
                if item is None:
                    continue
                path = str(item.data(Qt.ItemDataRole.UserRole) or "")
                if path:
                    mapping[path] = index
        self._thumbnail_path_to_row = mapping

    def _thumbnail_row_for_path(self: Any, path: str) -> int | None:
        normalized = self._normalized_path(path)
        row = getattr(self, "_thumbnail_path_to_row", {}).get(normalized)
        if row is not None:
            return int(row)
        if not hasattr(self, "thumbnail_grid"):
            return None
        for index in range(self.thumbnail_grid.count()):
            item = self.thumbnail_grid.item(index)
            if item is None:
                continue
            item_path = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if item_path and self._normalized_path(item_path) == normalized:
                return index
        return None

    def _make_thumbnail_grid_item(self: Any, path: str) -> QListWidgetItem:
        normalized = self._normalized_path(path)
        requested_size = self._thumbnail_request_size()
        cached_icon = getattr(self, "_thumbnail_icon_cache", {}).get(
            (normalized, requested_size)
        )
        item = QListWidgetItem(cached_icon or self._thumbnail_placeholder(), "")
        item.setSizeHint(self._thumbnail_icon_size)
        item.setToolTip(Path(str(path)).stem)
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        if cached_icon is not None:
            self._thumbnail_loaded_generation[normalized] = self._thumbnail_generation
            self._thumbnail_loaded_sizes[normalized] = requested_size
        frame_id = (
            self.thumbnail_grid.count()
            if hasattr(self, "thumbnail_grid")
            else len(getattr(self, "_thumbnail_path_to_row", {}))
        )
        item.setData(int(Qt.ItemDataRole.UserRole) + 1002, frame_id)
        if bool(getattr(self, "_asset_filter_match_only", False)) and not self._image_path_has_matching_vector(str(path)):
            item.setHidden(True)
        return item

    def _thumbnail_loading_blocked(self: Any) -> bool:
        store = getattr(self, "_pyramid_frame_store", None)
        if store is not None and store.has_lod():
            return True
        if not self._frame_matrix_enabled() or not self._frame_matrix_thumbnails_enabled():
            return True
        if bool(getattr(self, "_thumbnail_rebuild_in_progress", False)):
            return True
        if getattr(self, "_frame_load_running_path", None) is not None:
            return True
        return getattr(self, "_loading_image_path", None) is not None

    def _cancel_thumbnail_loading(self: Any) -> None:
        self._thumbnail_generation += 1
        self._thumbnail_queued_paths.clear()
        self._thumbnail_queued_sizes.clear()
        getattr(self, "_thumbnail_pending_apply", {}).clear()
        for timer in list(getattr(self, "_deferred_thumbnail_load_timers", [])):
            try:
                timer.stop()
            except RuntimeError:
                pass
        self._deferred_thumbnail_load_timers: list[QTimer] = []
        self._thumbnail_radial_paths: list[str] = []
        self._thumbnail_radial_cursor = 0
        self._thumbnail_radial_center_path = None
        if hasattr(self, "_thumbnail_radial_pump_timer"):
            self._thumbnail_radial_pump_timer.stop()
        if hasattr(self, "_thumbnail_apply_timer"):
            self._thumbnail_apply_timer.stop()
        if hasattr(self, "_thumbnail_visible_load_timer"):
            self._thumbnail_visible_load_timer.stop()
        if hasattr(self, "_thumbnail_scroll_settle_timer"):
            self._thumbnail_scroll_settle_timer.stop()
        try:
            self._thumbnail_thread_pool.clear()
        except AttributeError:
            pass

    def _pause_thumbnail_radial_fill(self: Any) -> None:
        self._thumbnail_generation += 1
        if hasattr(self, "_thumbnail_radial_pump_timer"):
            self._thumbnail_radial_pump_timer.stop()
        if hasattr(self, "_thumbnail_apply_timer"):
            self._thumbnail_apply_timer.stop()
        for timer in list(getattr(self, "_deferred_thumbnail_load_timers", [])):
            try:
                timer.stop()
            except RuntimeError:
                pass
        self._deferred_thumbnail_load_timers = []
        self._thumbnail_queued_paths.clear()
        self._thumbnail_queued_sizes.clear()
        getattr(self, "_thumbnail_pending_apply", {}).clear()
        try:
            self._thumbnail_thread_pool.clear()
        except AttributeError:
            pass

    def _thumbnail_grid_paths_in_radial_order(self: Any, center_row: int) -> list[str]:
        if not self._frame_matrix_enabled() or not hasattr(self, "thumbnail_grid"):
            return []
        columns = max(1, self._thumbnail_columns())
        center_r = center_row // columns
        center_c = center_row % columns
        ordered: list[tuple[int, int, str]] = []
        for index in range(self.thumbnail_grid.count()):
            item = self.thumbnail_grid.item(index)
            if item is None or item.isHidden():
                continue
            path = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if not path:
                continue
            row = index // columns
            col = index % columns
            ring = max(abs(row - center_r), abs(col - center_c))
            ordered.append((ring, index, path))
        ordered.sort(key=lambda entry: (entry[0], entry[1]))
        return [path for _, _, path in ordered]

    def _reseed_thumbnail_radial_fill(self: Any) -> None:
        if not self._frame_matrix_thumbnails_enabled():
            return
        if not hasattr(self, "thumbnail_grid") or self.thumbnail_grid.count() <= 0:
            return
        if self._thumbnail_loading_blocked():
            return
        # Follow the user's most recent selection immediately.  The workspace
        # changes only after the asynchronous frame load completes, so using it
        # alone leaves the matrix highlighting the previous frame meanwhile.
        current = getattr(self, "_desired_image_path", None) or self._workspace.current_image_path
        if not current:
            return
        normalized_current = self._normalized_path(current)
        path_index = getattr(self, "_thumbnail_path_to_row", {})
        center_row = path_index.get(normalized_current)
        if center_row is None:
            return
        loaded_generation = getattr(self, "_thumbnail_loaded_generation", {})
        loaded_sizes = getattr(self, "_thumbnail_loaded_sizes", {})
        queued_paths: set[str] = getattr(self, "_thumbnail_queued_paths", set())
        queued_sizes = getattr(self, "_thumbnail_queued_sizes", {})
        current_generation = self._thumbnail_generation
        requested_size = self._thumbnail_request_size()
        self._thumbnail_radial_paths = [
            path
            for path in self._thumbnail_grid_paths_in_radial_order(center_row)
            if (loaded_generation.get(path) != current_generation or loaded_sizes.get(path) != requested_size)
            and not (path in queued_paths and queued_sizes.get(path) == requested_size)
        ]
        self._thumbnail_radial_cursor = 0
        self._thumbnail_radial_center_path = normalized_current
        self._schedule_thumbnail_radial_pump()
