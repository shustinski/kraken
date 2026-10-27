"""processing_mixin: persist current overlay changes."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
)

from contour.widget_parts.processing_mixin_parts.module_state import _LOGGER


class PersistCurrentOverlayChangesMixin:
    def _persist_current_overlay_changes(self: Any) -> bool:
        """Persist editor polygons for the current frame (dataset export and/or linked CIF)."""
        from contour.widget_parts.processing_mixin import save_polygons_vector

        self._last_vector_persist_error = None
        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        if current_state is None or current_image_path is None:
            return True
        if self._editor_polygons_are_current_frame():
            current_polygons = self.get_polygons()
            self._workspace.update_current_polygons(current_polygons)
        else:
            current_polygons = [polygon.clone() for polygon in current_state.polygons]
        if not self._workspace.current_image_has_changes():
            self._update_frame_item_status(current_image_path)
            self._update_vector_edit_status_label()
            return True

        want_dataset = bool(self.dataset_mode_checkbox.isChecked())
        cif_path = current_state.loaded_cif_path or self._find_matching_cif_path(current_image_path)
        if cif_path and not current_state.loaded_cif_path:
            current_state.loaded_cif_path = str(cif_path)
        can_cif = bool(cif_path and current_state.source_image is not None)

        if not want_dataset and not can_cif:
            reason = self._tr(
                "vector_save_no_target_log",
                "Нет каталога набора данных или связанного CIF для сохранения правок текущего кадра."
                if self._ui_language == "ru"
                else "No dataset directory or linked CIF available to save edits for the current frame.",
            )
            self._last_vector_persist_error = reason
            self._append_log(reason)
            return False

        if want_dataset:
            saved_ds = self._export_dataset_frame_for_state(current_image_path, current_state, current_polygons)
            if not saved_ds:
                if not self._last_vector_persist_error:
                    self._last_vector_persist_error = self._tr(
                        "dataset_directory_not_set_log",
                        "Папка выборки не задана."
                        if self._ui_language == "ru"
                        else "Dataset directory is not set.",
                    )
                return False

        if can_cif:
            image_size = (int(current_state.source_image.shape[1]), int(current_state.source_image.shape[0]))
            try:
                if self._frame_switch_profiling_active():
                    from ...infrastructure.cif_operation_profiler import cif_operation_profiling

                    with cif_operation_profiling() as cif_timings:
                        save_polygons_vector(
                            cif_path,
                            current_image_path,
                            current_polygons,
                            image_size=image_size,
                        )
                    session = getattr(self, "_frame_switch_profile", None)
                    if session is not None:
                        session.merge_timings(cif_timings)
                else:
                    save_polygons_vector(
                        cif_path,
                        current_image_path,
                        current_polygons,
                        image_size=image_size,
                    )
                self._workspace.refresh_cif_revision(current_image_path, cif_path)
            except Exception as exc:
                _LOGGER.exception("Autosave failed for %s to %s", current_image_path, cif_path)
                reason = self._tr(
                    "autosave_failed_log",
                    "Не удалось сохранить CIF {path}: {error}"
                    if self._ui_language == "ru"
                    else "Failed to save CIF {path}: {error}",
                    path=cif_path,
                    error=exc,
                )
                self._last_vector_persist_error = reason
                self._append_log(reason)
                return False

        persisted_path = str(Path(current_image_path))
        self._persisted_highlight_paths.add(persisted_path)
        self._workspace.sync_polygon_reference_to_current(persisted_path)
        self._append_log(
            self._tr(
                "vectors_persisted_transition_log",
                "Изменения векторов сохранены для кадра {name}"
                if self._ui_language == "ru"
                else "Vector edits saved for frame {name}",
                name=Path(current_image_path).name,
            )
        )
        self._update_frame_item_status(current_image_path)
        self._update_vector_edit_status_label()
        return True
