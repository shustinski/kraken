"""processing_mixin: polygons edited."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
    QApplication,
    QEventLoop,
    QMessageBox,
    index_cif_file_paths,
)


class PolygonsEditedMixin:
    def _show_batch_progress(self: Any, total: int) -> None:
        self._batch_progress_enabled = True
        self._apply_batch_progress_bar(0, total)

    def _hide_batch_progress(self: Any) -> None:
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)

    def _shared_progress_job_active(self: Any) -> bool:
        return bool(
            self._batch_controller.is_running
            or getattr(self, "_antialias_running", False)
            or getattr(self, "_fix_internal_contours_running", False)
        )

    def _finish_shared_progress_if_idle(self: Any) -> None:
        if self._shared_progress_job_active():
            return
        self._batch_progress_enabled = False
        self._hide_batch_progress()

    def _apply_batch_progress_bar(
        self: Any,
        current: int,
        total: int,
        *,
        pump_events: bool = False,
    ) -> None:
        bar = self.batch_progress_bar
        maximum = max(1, int(total))
        value = max(0, min(int(current), maximum))
        bar.setRange(0, maximum)
        bar.setValue(value)
        bar.setVisible(True)
        bar.repaint()
        if pump_events:
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def _confirm_processing_question(self: Any, question: str) -> bool:
        parent = self.window() if hasattr(self, "window") else self
        answer = QMessageBox.question(
            parent,
            self._tr(
                "processing_confirm_title",
                "Обработка" if self._ui_language == "ru" else "Processing",
            ),
            question,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _collect_opened_cif_entries(self: Any) -> dict[str, str]:
        """Collect stem → CIF path mappings for batch vector tools."""

        entries: dict[str, str] = dict(self._workspace.cif_paths_by_stem)

        for _image_path, state in self._workspace.cached_states():
            loaded_cif_path = getattr(state, "loaded_cif_path", None)
            if not loaded_cif_path:
                continue
            cif_path = Path(str(loaded_cif_path))
            if cif_path.is_file() and cif_path.suffix.lower() in {".cif", ".cv"}:
                entries.setdefault(cif_path.stem.lower(), str(cif_path.resolve()))

        for image_path in self._workspace.image_paths:
            matched = self._find_matching_cif_path(image_path)
            if not matched:
                continue
            cif_path = Path(str(matched))
            if cif_path.is_file() and cif_path.suffix.lower() in {".cif", ".cv"}:
                entries.setdefault(cif_path.stem.lower(), str(cif_path.resolve()))

        if hasattr(self, "cif_dir_edit"):
            cif_directory = self.cif_dir_edit.text().strip()
            if cif_directory:
                directory = Path(cif_directory)
                if directory.is_dir():
                    vector_paths = [
                        str(path.resolve())
                        for path in directory.iterdir()
                        if path.is_file() and path.suffix.lower() in {".cif", ".cv"}
                    ]
                    entries.update(index_cif_file_paths(vector_paths))

        return entries

    def _confirm_and_start_batch_processing(self: Any) -> None:
        if not self._confirm_processing_question(
            self._tr(
                "batch_all_frames_confirm",
                "Запустить распознавание всех кадров?"
                if self._ui_language == "ru"
                else "Start recognition of all frames?",
            )
        ):
            return
        self.start_batch_processing()

    def _on_polygons_edited(self: Any) -> None:
        if self._updating_views:
            return
        layer_patch = None
        if hasattr(self, "view") and self.view is not None:
            layer_patch = self.view._editor_scene.take_last_workspace_layer_patch()
        if (
            layer_patch is not None
            and self._editor_polygons_are_current_frame()
        ):
            removed_ids, added = layer_patch
            if self._workspace.apply_layer_patch_to_current(removed_ids, added):
                current_path = self._workspace.current_image_path
                if current_path:
                    self._persisted_highlight_paths.discard(str(Path(current_path)))
                self._update_frame_item_status(self._workspace.current_image_path)
                self._update_vector_edit_status_label(sync_editor=False)
                self.polygonsEdited.emit()
                return
        changed_polygon_id = None
        if hasattr(self, "view") and self.view is not None:
            changed_polygon_id = self.view._editor_scene.take_last_geometry_changed_polygon_id()
        if (
            changed_polygon_id is not None
            and self._editor_polygons_are_current_frame()
            and hasattr(self, "view")
            and self.view is not None
        ):
            polygon = self.view._editor_scene._polygons.get(changed_polygon_id)
            if polygon is not None and self._workspace.update_polygon_in_current(changed_polygon_id, polygon):
                current_path = self._workspace.current_image_path
                if current_path:
                    self._persisted_highlight_paths.discard(str(Path(current_path)))
                self._update_frame_item_status(self._workspace.current_image_path)
                self._update_vector_edit_status_label(sync_editor=False)
                self.polygonsEdited.emit()
                return
        if self._sync_editor_polygons_to_current_workspace():
            current_path = self._workspace.current_image_path
            if current_path:
                self._persisted_highlight_paths.discard(str(Path(current_path)))
            self._update_frame_item_status(self._workspace.current_image_path)
            self._update_vector_edit_status_label(sync_editor=False)
            self.polygonsEdited.emit()
