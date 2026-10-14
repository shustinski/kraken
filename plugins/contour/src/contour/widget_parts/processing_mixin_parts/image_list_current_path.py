"""processing_mixin: image list current path."""
from __future__ import annotations
import shutil
from typing import Any
from contour.widget_parts._imports import (
    Path,
    QAbstractItemView,
    QModelIndex,
    Qt,
)


class ImageListCurrentPathMixin:
    def _thumbnail_disk_cache_marker_path(self: Any) -> Path:
        return Path(getattr(self, "_thumbnail_disk_cache_dir", Path())) / "cache.key"

    def _thumbnail_disk_cache_key_for_base_paths(self: Any, paths: list[str]) -> str:
        if not paths:
            return ""
        return str(Path(paths[0]).parent)

    def _clear_thumbnail_disk_cache(self: Any) -> None:
        if hasattr(self, "_cancel_thumbnail_loading"):
            self._cancel_thumbnail_loading()
        cache_dir = Path(getattr(self, "_thumbnail_disk_cache_dir", Path()))
        if not cache_dir:
            return
        try:
            shutil.rmtree(cache_dir)
        except FileNotFoundError:
            pass
        except Exception:
            return
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._thumbnail_loaded_generation.clear()
        self._thumbnail_loaded_sizes.clear()
        self._thumbnail_queued_paths.clear()
        self._thumbnail_queued_sizes.clear()
        getattr(self, "_thumbnail_icon_cache", {}).clear()
        getattr(self, "_thumbnail_pending_apply", {}).clear()

    def _reset_thumbnail_disk_cache_for_base_paths(self: Any, paths: list[str], *, force: bool = False) -> None:
        cache_dir = Path(getattr(self, "_thumbnail_disk_cache_dir", Path()))
        if not cache_dir:
            return
        key = self._thumbnail_disk_cache_key_for_base_paths(paths)
        marker_path = self._thumbnail_disk_cache_marker_path()
        previous_key = ""
        try:
            previous_key = marker_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            previous_key = ""
        except Exception:
            previous_key = ""
        if force or (previous_key and previous_key != key):
            self._clear_thumbnail_disk_cache()
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            marker_path.write_text(key, encoding="utf-8")
        except Exception:
            pass
        self._thumbnail_disk_cache_key = key

    def _image_list_source_index(self: Any, proxy_index: QModelIndex) -> QModelIndex:
        if not proxy_index.isValid():
            return QModelIndex()
        return self._image_list_proxy.mapToSource(proxy_index)

    def _image_list_path_from_proxy_index(self: Any, proxy_index: QModelIndex) -> str | None:
        source_index = self._image_list_source_index(proxy_index)
        if not source_index.isValid():
            return None
        value = self._image_list_model.data(source_index, Qt.ItemDataRole.UserRole)
        return str(value) if value else None

    def _image_list_selected_paths(self: Any) -> list[str]:
        paths: list[str] = []
        selection = self.image_list.selectionModel()
        if selection is None:
            return paths
        for proxy_index in selection.selectedIndexes():
            path = self._image_list_path_from_proxy_index(proxy_index)
            if path:
                paths.append(path)
        return paths

    def _sync_workspace_selection_to_path(self: Any, image_path: str) -> None:
        normalized = str(Path(image_path))
        self._workspace._current_image_path = normalized
        # Selection is only a requested destination.  Keeping a cached state
        # here makes resolve_cached_load() report "already current" before the
        # editor has actually switched to this frame.
        self._workspace._current_state = None

    def _set_image_list_current_path(self: Any, image_path: str | None, *, fallback_to_first: bool = True) -> bool:
        if image_path:
            row = self._image_path_index(image_path)
            if row is not None:
                proxy_index = self._image_list_proxy.mapFromSource(self._image_list_model.index(row))
                if proxy_index.isValid():
                    self._set_image_list_current_index_preserving_scroll(proxy_index)
                    selected_path = self._image_list_path_from_proxy_index(self.image_list.currentIndex())
                    if str(Path(selected_path or "")) == str(Path(image_path)):
                        self._sync_workspace_selection_to_path(image_path)
                        return True
                    return False
                return False
        if fallback_to_first and self._image_list_proxy.rowCount() > 0:
            self._set_image_list_current_index_preserving_scroll(self._image_list_proxy.index(0, 0))
            selected_path = self._image_list_path_from_proxy_index(self.image_list.currentIndex())
            if selected_path:
                self._sync_workspace_selection_to_path(selected_path)
                return True
            return self.image_list.currentIndex().isValid()
        elif self._image_list_proxy.rowCount() <= 0:
            self._sync_current_state_views()
        return False

    def _set_image_list_current_index_preserving_scroll(self: Any, proxy_index: QModelIndex) -> None:
        """Select a row without jumping the list when the row is already on screen."""

        view = self.image_list
        bar = view.verticalScrollBar()
        previous = int(bar.value()) if bar is not None else 0
        viewport = view.viewport()
        already_visible = False
        if viewport is not None and proxy_index.isValid():
            row_rect = view.visualRect(proxy_index)
            already_visible = viewport.rect().intersects(row_rect) and not row_rect.isEmpty()
        view.setCurrentIndex(proxy_index)
        if already_visible and bar is not None and int(bar.value()) != previous:
            bar.setValue(previous)
        elif not already_visible and proxy_index.isValid():
            view.scrollTo(proxy_index, QAbstractItemView.ScrollHint.EnsureVisible)

    def _image_path_in_image_list(self: Any, image_path: str) -> bool:
        return self._image_path_index(image_path) is not None

    def _update_frame_item_status(self: Any, image_path: str | None) -> None:
        if not image_path:
            return
        self._image_list_model.invalidate_path(image_path)
        self._refresh_vector_items_for_stems({Path(str(image_path)).stem.lower()})
        self._update_thumbnail_item_status(image_path)
        self._update_asset_items_for_image_path(image_path)

    def _refresh_image_list_item_states(self: Any) -> None:
        if self._uses_large_frame_list():
            current = self._workspace.current_image_path
            if current:
                self._image_list_model.invalidate_path(current)
        else:
            self._image_list_model.invalidate_all_rows()
        self._refresh_vector_rows_for_workspace()
        self._refresh_asset_filter_list_item_states()
        self._update_thumbnail_grid_selection()

    def _update_thumbnail_item_status(self: Any, image_path: str | None) -> None:
        if not image_path or not self._frame_matrix_enabled() or not hasattr(self, "thumbnail_grid"):
            return
        normalized = str(Path(image_path))
        row = self._thumbnail_row_for_path(normalized) if hasattr(self, "_thumbnail_row_for_path") else None
        if row is None:
            return
        item = self.thumbnail_grid.item(row)
        if item is None:
            return
        self._paint_image_row_item(item, normalized, show_text=False)

    def _update_vector_edit_status_label(self: Any, *, sync_editor: bool = True) -> None:
        if not hasattr(self, "vector_edit_status_label"):
            return
        if self._workspace.current_image_path is None:
            self.vector_edit_status_label.clear()
            return
        if sync_editor and not self._updating_views:
            self._sync_editor_polygons_to_current_workspace()
        current_path = str(Path(self._workspace.current_image_path))
        dirty = (
            self._workspace.current_image_has_changes()
            or current_path in self._restored_modified_image_paths
        )
        if self._ui_language == "ru":
            self.vector_edit_status_label.setText("Изменено" if dirty else "Сохранено")
        else:
            self.vector_edit_status_label.setText("Modified" if dirty else "Saved")
