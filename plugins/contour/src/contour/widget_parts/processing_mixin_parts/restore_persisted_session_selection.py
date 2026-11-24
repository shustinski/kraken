"""processing_mixin: restore persisted session selection."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    ASSET_FILTER_LISTS_MAX_FRAMES,
    Path,
    QListWidgetItem,
    Qt,
)


class RestorePersistedSessionSelectionMixin:
    def _remember_produced_recognition(self: Any, image_path: str, polygons: list) -> None:
        catalog = getattr(self, "_produced_recognition", None)
        if catalog is None:
            return
        catalog.remember(image_path, polygons)

    def _show_produced_recognition_for_current_file(self: Any, image_path: str) -> None:
        from ...application.recognition_display import install_produced_recognition_for_switch

        catalog = getattr(self, "_produced_recognition", None)
        if catalog is None:
            return
        install_produced_recognition_for_switch(self._workspace, catalog, image_path)

    def _on_batch_result(self: Any, result) -> None:
        polygons = list(getattr(result, "polygons", []) or [])
        if not getattr(result, "error", None):
            self._remember_produced_recognition(result.image_path, polygons)
            current = self._workspace.current_image_path
            if current and str(Path(current)) == str(Path(result.image_path)):
                self._show_produced_recognition_for_current_file(result.image_path)
                if hasattr(self, "_sync_current_state_views"):
                    self._sync_current_state_views(preserve_view=True, sync_neighbors=False)
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
        """Rescan folder sources and drop file-source frames that vanished on disk."""

        from ...application.sources import SOURCE_IMAGES, SOURCE_VECTORS
        from dataclasses import replace as _replace

        registry = self.source_registry
        if registry.is_empty():
            self._append_log(self._tr("input_directory_empty_log"))
            return
        for source in registry.sources():
            if source.locked:
                continue
            if source.is_folder:
                registry.update(_replace(source, scanned=False))
                self._queue_source_scan(registry.get(source.id))
            else:
                existing = tuple(path for path in source.paths if Path(path).is_file())
                if existing != source.paths:
                    registry.update(_replace(source, paths=existing))
        self._apply_sources(SOURCE_IMAGES)
        self._apply_sources(SOURCE_VECTORS)

    def set_input_directory(self: Any, path: str) -> None:
        """Load one frame folder instead of all current frame sources (CLI, Kraken)."""

        from ...application.sources import SOURCE_IMAGES, folder_source

        directory = self._path_settings.validate_input_directory(path)
        if not directory.available:
            self._append_log(
                self._tr(
                    "input_directory_missing_log",
                    directory=directory.path,
                )
            )
            return
        self._remember_recent_path("input", directory.path)
        self.replace_sources(SOURCE_IMAGES, folder_source(SOURCE_IMAGES, directory.path))

    def set_cif_directory(self: Any, path: str) -> None:
        """Load one vector folder instead of all current vector sources (CLI, Kraken)."""

        from ...application.sources import SOURCE_VECTORS, folder_source

        directory = str(Path(path))
        self._remember_recent_path("cif", directory)
        self.replace_sources(SOURCE_VECTORS, folder_source(SOURCE_VECTORS, directory))

    def set_output_directory(self: Any, path: str) -> None:
        self.output_dir_edit.setText(path)
        self._save_persisted_paths()
        self._remember_recent_path("output", path)
        self._schedule_path_card_status_refresh()

    def set_dataset_directory(self: Any, path: str) -> None:
        self.dataset_dir_edit.setText(path)
        self._save_persisted_paths()
        self._remember_recent_path("dataset", path)
        self._schedule_path_card_status_refresh()

    def _restore_persisted_session_selection(self: Any) -> None:
        if not hasattr(self, "_session_settings_store"):
            return
        self._restore_sources_from_settings()
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
        # Frames must not load "no vectors" while a vector folder is still indexing.
        self._defer_vector_load_until_cif_index = bool(self._vector_scan_pending())
        pending = getattr(self, "_pending_image_list_post_build", None)
        if pending:
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
        if not self._apply_vector_sources_if_dirty():
            self._start_vector_index_or_rebuild_frame_matrix_after_images()

    def _select_loaded_image_path(self: Any, image_path: str | None, *, fallback_to_first: bool = True) -> None:
        self._set_image_list_current_path(image_path, fallback_to_first=fallback_to_first)
