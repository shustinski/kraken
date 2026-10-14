"""navigation_mixin: sync after cif index."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
    QFileDialog,
    QListWidgetItem,
    QSignalBlocker,
    is_image_path,
    time,
)

from contour.widget_parts.navigation_mixin_parts.module_state import _LOGGER


class SyncAfterCifIndexMixin:
    def _sync_after_cif_index_changed(self: Any) -> None:
        self._clear_cif_transient_hints()
        self._refresh_image_list_item_states()
        desired = str(getattr(self, "_desired_image_path", "") or "")
        cur = (
            desired
            or self._workspace.current_image_path
            or str(getattr(self, "_loading_image_path", "") or "")
            or str(getattr(self, "_frame_load_running_path", "") or "")
        )
        if cur:
            state = self._workspace.current_state
            state_matches_target = (
                state is not None
                and str(Path(state.image_path)) == str(Path(cur))
            )
            if state_matches_target and state.source_image is not None:
                try:
                    self._reload_current_frame_vectors()
                except Exception as exc:
                    _LOGGER.exception("Failed to reload vectors for %s", cur)
                    self._append_log(self._tr("reload_with_cif_failed_log", error=exc))
            else:
                try:
                    self.load_image(cur, load_vectors=True)
                except Exception as exc:
                    _LOGGER.exception("Failed to reload image and vectors for %s", cur)
                    self._append_log(self._tr("reload_with_cif_failed_log", error=exc))
        if not self._uses_large_frame_list():
            report = self._matching_report()
            self._log_matching_gaps_after_refresh(report)

    def _clear_cif_transient_hints(self: Any) -> None:
        self._cif_load_failure_stems.clear()

    def _invalidate_cif_overlay_for_stems(self: Any, stems: set[str]) -> list[str]:
        if not stems:
            return []
        paths: list[str] = []
        cif_paths: list[str] = []
        for path in self._workspace.image_paths:
            stem = Path(path).stem.lower()
            if stem in stems:
                self._cif_load_failure_stems.discard(stem)
                paths.append(str(Path(path)))
                cif_path = self._workspace.resolve_cif_path(path)
                if cif_path:
                    cif_paths.append(cif_path)
        if paths:
            self._workspace.forget_cleared_vectors(paths)
            self._workspace.invalidate_image_states(paths)
            if hasattr(self, "_neighbor_vector_cache"):
                for path in paths:
                    self._neighbor_vector_cache.pop(str(Path(path)), None)
        if cif_paths:
            from contour.serializers import invalidate_cif_parse_cache

            invalidate_cif_parse_cache(cif_paths)
        return paths

    def _reload_cif_overlays_for_selected_vectors(self: Any) -> None:
        stems: set[str] = set()
        for row in self.vector_list.selectedItems():
            tip = row.toolTip()
            if tip:
                stems.add(Path(tip).stem.lower())
        if not stems:
            self._append_log(self._tr("no_vector_selection_for_reload_log"))
            return
        affected = self._invalidate_cif_overlay_for_stems(stems)
        self._finalize_overlay_reload(affected)

    def _reload_cif_overlays_for_selected_images(self: Any) -> None:
        stems: set[str] = {Path(path).stem.lower() for path in self._image_list_selected_paths()}
        cur = self._workspace.current_image_path
        if not stems and cur:
            stems.add(Path(cur).stem.lower())
        if not stems:
            self._append_log(self._tr("no_image_selection_for_reload_log"))
            return
        affected = self._invalidate_cif_overlay_for_stems(stems)
        self._finalize_overlay_reload(affected)

    def _finalize_overlay_reload(self: Any, paths_for_message: list[str]) -> None:
        if not paths_for_message:
            self._append_log(self._tr("cif_reload_no_matching_images_log"))
            return
        unique_stems = sorted({Path(p).stem.lower() for p in paths_for_message})[:16]
        more_txt = ""
        extra = len({Path(p).stem.lower() for p in paths_for_message}) - len(unique_stems)
        if extra > 0:
            more_txt = f" (+{extra})"
        self._append_log(
            self._tr(
                "cif_reload_invalidate_log",
                stems=", ".join(unique_stems),
                more=more_txt,
            )
        )
        self._refresh_image_list_item_states()
        self._refresh_vector_rows_for_workspace()
        cur = self._workspace.current_image_path
        if cur and cur in paths_for_message:
            try:
                self.load_image(cur)
            except Exception as exc:
                _LOGGER.exception("Failed to reload image %s after vector refresh", cur)
                self._append_log(self._tr("reload_with_cif_failed_log", error=exc))

    def _on_vector_item_navigate_request(self: Any, item: QListWidgetItem) -> None:
        """Jump to the matching image frame (Files → Images) after an explicit click."""

        if item is None:
            return
        if time.monotonic() < self._vectors_list_ignore_navigate_until:
            return
        tip = item.toolTip()
        if not tip:
            return
        stem = Path(tip).stem.lower()
        image_path = self._image_path_for_cif_stem(stem)
        if not image_path:
            self._append_log(self._tr("vector_row_no_matching_image_loaded_log"))
            return
        if not self._image_path_in_image_list(image_path):
            return
        if hasattr(self, "sidebar_list_mode_combo"):
            with QSignalBlocker(self.sidebar_list_mode_combo):
                self.sidebar_list_mode_combo.setCurrentIndex(0)
        if hasattr(self, "sidebar_list_stack"):
            self.sidebar_list_stack.setCurrentIndex(0)
        self._set_image_list_current_path(image_path, fallback_to_first=False)

    def _select_input_directory(self: Any) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_input_directory_dialog"),
            self._dialog_start_directory_from_line_edit(self.input_dir_edit),
        )
        if not path:
            return
        directory = self._path_settings.validate_input_directory(path)
        if not directory.available:
            self._append_log(self._tr("input_directory_missing_log", directory=directory.path))
            return
        self.set_input_directory(directory.path)

    def _select_cif_directory(self: Any) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_cif_directory_dialog"),
            self._dialog_start_directory_from_line_edit(self.cif_dir_edit),
        )
        if path:
            self.set_cif_directory(path)

    def _select_output_directory(self: Any) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_output_directory_dialog"),
            self._dialog_start_directory_from_line_edit(self.output_dir_edit),
        )
        if path:
            self.set_output_directory(path)

    def _select_dataset_directory(self: Any) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_dataset_directory_dialog"),
            self._dialog_start_directory_from_line_edit(self.dataset_dir_edit),
        )
        if path:
            self.set_dataset_directory(path)

    def _apply_input_directory_edit(self: Any) -> None:
        path = self.input_dir_edit.text().strip()
        if path:
            self.set_input_directory(path)
        else:
            self._workspace.replace_image_selection([], is_supported_image=is_image_path)
            self._base_frame_number_by_path: dict[str, int] = {}
            self._base_frame_numbers: set[int] = set()
            self._set_image_list_paths([])
            self._rebuild_thumbnail_grid()
            self._clear_extra_layers()
            self._update_extra_layers_enabled_state()
            self._rebuild_vector_list()
            self._refresh_image_list_item_states()
            self._sync_current_state_views()
            self._save_persisted_paths()
