"""processing_mixin: restore persisted session selection."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    ASSET_FILTER_LISTS_MAX_FRAMES,
    Path,
    QListWidgetItem,
    Qt,
    index_cif_file_paths,
    is_image_path,
)


class RestorePersistedSessionSelectionMixin:
    def _on_batch_result(self: Any, result) -> None:
        polygons = list(getattr(result, "polygons", []) or [])
        # Multiprocessing batch workers return metadata only to keep IPC and the
        # GUI event queue bounded during multi-thousand-image runs.
        self.imageProcessed.emit(result.image_path, polygons)
        self._append_log(
            self._tr(
                "batch_result_log",
                image_name=Path(result.image_path).name,
                count=int(getattr(result, "polygon_count", len(polygons))),
            )
        )

    def _on_batch_progress(self: Any, current: int, total: int) -> None:
        if self._batch_progress_enabled:
            self._apply_batch_progress_bar(current, total)
        self._set_progress_status("batch_progress_status", current=current, total=total)
        self.batchProgress.emit(current, total)

    def _on_batch_finished(self: Any) -> None:
        self._finish_shared_progress_if_idle()
        self._set_progress_status("batch_finished_status")
        self.batchFinished.emit()

    def _on_batch_error(self: Any, image_path: str, message: str) -> None:
        self._append_log(self._tr("batch_error_log", image_name=Path(image_path).name, message=message))

    def refresh_image_list(self: Any) -> None:
        from ...infrastructure.settings_store import IMAGE_LIST_MODE_EXPLICIT

        if getattr(self, "_image_list_mode", None) == IMAGE_LIST_MODE_EXPLICIT:
            self._revalidate_explicit_image_paths()
            return
        directory = self.input_dir_edit.text().strip()
        if not directory:
            self._append_log(self._tr("input_directory_empty_log"))
            return
        self._begin_async_directory_scan(directory)

    def _revalidate_explicit_image_paths(self: Any) -> None:
        existing_paths = [
            str(Path(path))
            for path in self._workspace.image_paths
            if (str(Path(path)) in getattr(self, "_source_frame_numbers", {}) or Path(path).is_file()) and is_image_path(path)
        ]
        preferred = self._workspace.current_image_path
        self.load_images(existing_paths, preferred_current_image_path=preferred)

    def set_input_directory(self: Any, path: str) -> None:
        from ...infrastructure.settings_store import IMAGE_LIST_MODE_DIRECTORY

        directory = self._path_settings.validate_input_directory(path)
        if not directory.available:
            self._append_log(
                self._tr(
                    "input_directory_missing_log",
                    directory=directory.path,
                )
            )
            return
        self.input_dir_edit.setText(directory.path)
        self._save_persisted_paths()
        self._set_image_list_mode(IMAGE_LIST_MODE_DIRECTORY)
        self._begin_async_directory_scan(directory.path)

    def set_cif_directory(self: Any, path: str) -> None:
        directory = str(Path(path))
        self.cif_dir_edit.setText(directory)
        self._save_persisted_paths()
        self._indexed_cif_directory = None
        if bool(getattr(self, "_image_list_rebuild_in_progress", False)):
            self._pending_cif_directory_path_after_images = directory
            return
        self._begin_async_cif_directory_index(directory)

    def set_output_directory(self: Any, path: str) -> None:
        self.output_dir_edit.setText(path)
        self._save_persisted_paths()

    def set_dataset_directory(self: Any, path: str) -> None:
        self.dataset_dir_edit.setText(path)
        self._save_persisted_paths()

    def _restore_persisted_session_selection(self: Any) -> None:
        if not hasattr(self, "_session_settings_store"):
            return
        from ...infrastructure.settings_store import IMAGE_LIST_MODE_DIRECTORY

        image_paths = [
            str(Path(path))
            for path in self._session_settings_store.load_image_paths()
            if Path(path).is_file() and is_image_path(path)
        ]
        vector_paths = [
            str(Path(path))
            for path in self._session_settings_store.load_vector_paths()
            if Path(path).is_file()
        ]
        if vector_paths:
            self._pending_restore_vector_paths = list(vector_paths)
            self._workspace.set_cif_index(index_cif_file_paths(vector_paths))
            self._rebuild_vector_list()
        self._image_list_mode = self._session_settings_store.load_image_list_mode()
        preferred = self._session_settings_store.load_current_image_path()
        if preferred:
            preferred = str(Path(preferred))
        self._pending_restore_current_image_path = preferred
        input_directory = self.input_dir_edit.text().strip() if hasattr(self, "input_dir_edit") else ""
        if (
            self._image_list_mode == IMAGE_LIST_MODE_DIRECTORY
            and input_directory
            and Path(input_directory).is_dir()
        ):
            cif_directory = self.cif_dir_edit.text().strip() if hasattr(self, "cif_dir_edit") else ""
            self._pending_cif_directory_path_after_images = (
                str(Path(cif_directory))
                if cif_directory and Path(cif_directory).is_dir()
                else None
            )
            self._begin_async_directory_scan(str(Path(input_directory)))
            return
        if not image_paths:
            self._refresh_image_list_item_states()
            return
        self.load_images(image_paths, preferred_current_image_path=preferred)
        self._refresh_image_list_item_states()

    def _rebuild_image_list_items(self: Any, normalized_paths: list[str]) -> None:
        self._rebuild_image_list_items_responsive(normalized_paths)
        return
        self.image_list.clear()
        for path in normalized_paths:
            item = QListWidgetItem(Path(path).stem)
            item.setToolTip(f"Путь к файлу: {path}" if self._ui_language == "ru" else f"File path: {path}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._paint_image_row_item(item, path)
            self.image_list.addItem(item)
        self._rebuild_thumbnail_grid()

    def _rebuild_image_list_items_responsive(self: Any, normalized_paths: list[str]) -> None:
        self._image_list_build_generation += 1
        self._image_list_rebuild_in_progress = True
        paths = [str(Path(path)) for path in normalized_paths]
        if hasattr(self, "image_vector_list") and len(paths) <= ASSET_FILTER_LISTS_MAX_FRAMES:
            for list_widget in (self.image_vector_list, self.image_only_list, self.vector_only_list):
                list_widget.clear()
        if hasattr(self, "files_scan_progress_bar"):
            self.files_scan_progress_bar.setVisible(False)
            self.files_scan_progress_bar.setRange(0, 100)
            self.files_scan_progress_bar.setValue(0)
        self._set_image_list_paths(paths)
        self._rebuild_asset_filter_lists()
        if len(paths) <= ASSET_FILTER_LISTS_MAX_FRAMES:
            self._apply_asset_view_filter()
        self._finish_pending_image_list_rebuild()

    def _finish_pending_image_list_rebuild(self: Any) -> None:
        self._image_list_rebuild_in_progress = False
        directory = self.cif_dir_edit.text().strip() if hasattr(self, "cif_dir_edit") else ""
        normalized_directory = str(Path(directory)) if directory else ""
        self._defer_vector_load_until_cif_index = bool(
            normalized_directory and getattr(self, "_indexed_cif_directory", None) != normalized_directory
        )
        pending = getattr(self, "_pending_image_list_post_build", None)
        if not pending:
            if self._defer_vector_load_until_cif_index:
                self._mark_thumbnail_grid_rebuild_pending()
            if not self._apply_pending_cif_directory_state_after_image_rebuild():
                self._start_vector_index_or_rebuild_frame_matrix_after_images()
            return
        self._pending_image_list_post_build = None
        select_path = pending.get("select_path")
        self._select_loaded_image_path(
            str(select_path) if select_path else None,
            fallback_to_first=bool(pending.get("fallback_to_first", True)),
        )
        self._refresh_vector_rows_for_workspace()
        if not self._uses_large_frame_list():
            self._log_matching_gaps_after_refresh(self._matching_report())
        if self._defer_vector_load_until_cif_index:
            self._mark_thumbnail_grid_rebuild_pending()
        if not self._apply_pending_cif_directory_state_after_image_rebuild():
            self._start_vector_index_or_rebuild_frame_matrix_after_images()

    def _select_loaded_image_path(self: Any, image_path: str | None, *, fallback_to_first: bool = True) -> None:
        self._set_image_list_current_path(image_path, fallback_to_first=fallback_to_first)
