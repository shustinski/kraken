"""navigation_mixin: activate frame path matrix."""
from __future__ import annotations
from typing import Any
from contour.value_conversion import to_int
from contour.widget_parts._imports import (
    LARGE_FRAME_COUNT_THRESHOLD,
    Path,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPoint,
    Qt,
    classify_dropped_paths,
    index_cif_file_paths,
)

from contour.widget_parts.navigation_mixin_parts.module_state import _LOGGER


class ActivateFramePathMatrixMixin:
    def _on_thumbnail_item_clicked(self: Any, item: QListWidgetItem) -> None:
        if item is None:
            return
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not path:
            return
        self._activate_frame_path_from_matrix(path)

    def _on_image_list_context_menu(self: Any, position: QPoint) -> None:
        index = self.image_list.indexAt(position)
        path = self._image_list_path_from_proxy_index(index)
        if not path:
            return
        global_pos = self.image_list.viewport().mapToGlobal(position)
        self._show_frame_context_menu(path, global_pos)

    def _on_asset_image_list_context_menu(self: Any, list_widget: QListWidget, position: QPoint) -> None:
        item = list_widget.itemAt(position)
        if item is None:
            return
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not path:
            return
        viewport = list_widget.viewport()
        assert viewport is not None
        global_pos = viewport.mapToGlobal(position)
        self._show_frame_context_menu(path, global_pos)

    def _on_frame_matrix_context_menu(self: Any, frame_id: object, global_pos: QPoint) -> None:
        try:
            index = to_int(frame_id)
        except (TypeError, ValueError):
            return
        image_paths = [str(Path(path)) for path in getattr(self._workspace, "image_paths", [])]
        if index < 0 or index >= len(image_paths):
            return
        self._show_frame_context_menu(image_paths[index], global_pos)

    def _show_frame_context_menu(self: Any, image_path: str, global_pos: QPoint) -> None:
        menu = QMenu(self)
        delete_action = menu.addAction(self._tr("delete_frame_action"))
        clear_action = menu.addAction(self._tr("clear_frame_action"))
        chosen = menu.exec(global_pos)
        if chosen is delete_action:
            self.remove_images([image_path])
        elif chosen is clear_action:
            self.clear_frame_vectors(image_path)

    def _ingest_dropped_paths(self: Any, paths: list[str]) -> None:
        payload = classify_dropped_paths(paths)
        if payload.is_empty():
            return
        if payload.image_paths:
            self.append_images(list(payload.image_paths), select_first_new=True)
        if payload.vector_paths:
            self._merge_cif_file_paths(list(payload.vector_paths))

    def _activate_frame_path_from_matrix(self: Any, path: str) -> None:
        normalized_path = str(Path(path))
        self._suppress_thumbnail_grid_scroll_path = normalized_path
        has_image_list_rows = bool(
            getattr(self, "_image_list_proxy", None) is not None and self._image_list_proxy.rowCount() > 0
        )
        if has_image_list_rows:
            if not self._set_image_list_current_path(normalized_path, fallback_to_first=False):
                self._suppress_thumbnail_grid_scroll_path = None
                self._update_thumbnail_grid_selection(scroll_to_selection=False)
                return
            current_index = self.image_list.currentIndex()
            self._on_image_list_clicked(current_index)
            return
        if self._workspace.current_state is not None and not self._try_leave_current_frame():
            self._suppress_thumbnail_grid_scroll_path = None
            self._update_thumbnail_grid_selection(scroll_to_selection=False)
            return
        self._workspace._current_image_path = normalized_path
        self._workspace._current_state = None
        try:
            self.load_image(normalized_path, preserve_editor_view_position=True)
        except Exception as exc:
            _LOGGER.exception("Failed to load image %s", normalized_path)
            self._append_log(self._tr("failed_to_load_image_log", image_path=normalized_path, error=exc))
            QMessageBox.warning(self, self._tr("image_load_error_title"), str(exc))

    def _on_frame_navigation_requested(self: Any, frame_id: object) -> None:
        try:
            index = to_int(frame_id)
        except (TypeError, ValueError):
            return
        image_paths = [str(Path(path)) for path in getattr(self._workspace, "image_paths", [])]
        if index < 0 or index >= len(image_paths):
            return
        path = image_paths[index]
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_current_frame_id(index, center=False, emit_signal=False)
        self._activate_frame_path_from_matrix(path)

    def _on_editor_current_frame_changed(self: Any, frame_id: object) -> None:
        try:
            index = to_int(frame_id)
        except (TypeError, ValueError):
            return
        image_paths = [str(Path(path)) for path in getattr(self._workspace, "image_paths", [])]
        if index < 0 or index >= len(image_paths):
            return
        current = str(Path(getattr(self._workspace, "current_image_path", "") or ""))
        if current == image_paths[index]:
            self._update_thumbnail_grid_selection(scroll_to_selection=False)
            return
        self._set_image_list_current_path(image_paths[index], fallback_to_first=False)

    def _refresh_vector_rows_for_workspace(self: Any) -> None:
        if not hasattr(self, "vector_list"):
            return
        if len(self._workspace.cif_paths_by_stem) > LARGE_FRAME_COUNT_THRESHOLD:
            return
        for index in range(self.vector_list.count()):
            row = self.vector_list.item(index)
            if row is None:
                continue
            tip = row.toolTip()
            if not tip:
                continue
            self._paint_vector_list_item(row, Path(tip).stem.lower())

    def _refresh_vector_items_for_stems(self: Any, stems: set[str]) -> None:
        if not stems or not hasattr(self, "vector_list"):
            return
        lowered = {s.lower() for s in stems}
        for idx in range(self.vector_list.count()):
            item = self.vector_list.item(idx)
            if item is None:
                continue
            tip = item.toolTip()
            if not tip:
                continue
            stem = Path(tip).stem.lower()
            if stem in lowered:
                self._paint_vector_list_item(item, stem)

    def _select_input_image_files(self: Any) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            self._tr("select_image_files_dialog"),
            self._dialog_start_directory_from_line_edit(self.input_dir_edit),
            self._tr("supported_image_files_filter"),
        )
        if not paths:
            return
        self.input_dir_edit.setText(str(Path(paths[0]).parent))
        self._save_persisted_paths()
        self.append_images([str(Path(p)) for p in paths])

    def _merge_cif_files_dialog(self: Any) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            self._tr("merge_cif_files_dialog"),
            self._dialog_start_directory_from_line_edit(self.cif_dir_edit),
            self._tr("cif_files_filter"),
        )
        if not paths:
            return
        self._merge_cif_file_paths([str(Path(path)) for path in paths])

    def _merge_cif_file_paths(self: Any, paths: list[str]) -> None:
        additions = index_cif_file_paths(paths)
        if not additions:
            return
        self._workspace.merge_cif_paths(additions)
        self._workspace.forget_cleared_vectors(
            [path for path in self._workspace.image_paths if Path(path).stem.lower() in additions]
        )
        self.cif_dir_edit.setText(str(Path(paths[0]).parent))
        self._save_persisted_paths()
        self._append_log(self._tr("cif_indexed_log", count=len(self._workspace.cif_paths_by_stem)))
        self._sync_after_cif_index_changed()
