"""processing_mixin: antialias finished."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    AntialiasCifItemResult,
    AntialiasCifJobSummary,
    Path,
)


class AntialiasFinishedMixin:
    def _sync_antialias_button_enabled(self: Any) -> None:
        if not hasattr(self, "antialias_opened_cif_button"):
            return
        edits_locked = bool(
            hasattr(self, "polygon_editor") and self.polygon_editor.vector_edits_locked()
        )
        self.antialias_opened_cif_button.setEnabled(
            not edits_locked and not bool(getattr(self, "_antialias_running", False))
        )

    def _cancel_antialias_job(self: Any) -> None:
        cancel_event = getattr(self, "_antialias_cancel", None)
        if cancel_event is not None:
            cancel_event.set()

    def _cancel_fix_internal_contours_job(self: Any) -> None:
        cancel_event = getattr(self, "_fix_internal_contours_cancel", None)
        if cancel_event is not None:
            cancel_event.set()

    def _on_antialias_progress(self: Any, run_id: int, current: int, total: int) -> None:
        if run_id != getattr(self, "_antialias_run_id", 0):
            return
        if self._batch_progress_enabled:
            self._apply_batch_progress_bar(current, total)
        self._set_progress_status("batch_progress_status", current=current, total=total)

    def _on_antialias_item_finished(self: Any, result: AntialiasCifItemResult) -> None:
        if getattr(self, "_closing", False):
            return
        if int(result.run_id) != getattr(self, "_antialias_run_id", 0):
            return
        if result.error or not result.changed:
            return
        image_path = result.image_path
        image_key = str(Path(image_path)) if image_path else None
        current_path = self._workspace.current_image_path
        current_key = str(Path(current_path)) if current_path else None
        antialiased = [polygon.clone() for polygon in result.polygons]
        if image_key:
            for cached_path, state in self._workspace.cached_states():
                if str(Path(cached_path)) != image_key:
                    continue
                state.polygons = [polygon.clone() for polygon in antialiased]
                state.reference_polygons = [polygon.clone() for polygon in antialiased]
                state.polygons_dirty = False
                state.loaded_cif_path = str(result.cif_path)
                break
            self._persisted_highlight_paths.add(image_key)
            self._update_frame_item_status(image_path)
            neighbor_cache = getattr(self, "_neighbor_vector_cache", None)
            if isinstance(neighbor_cache, dict):
                neighbor_cache.pop(image_key, None)
        if image_key is not None and image_key == current_key and hasattr(self, "polygon_editor"):
            self._updating_views = True
            try:
                self.polygon_editor.set_polygons(antialiased)
            finally:
                self._updating_views = False
            self._update_vector_edit_status_label()

    def _on_antialias_finished(self: Any, summary: AntialiasCifJobSummary) -> None:
        if int(summary.run_id) != getattr(self, "_antialias_run_id", 0):
            return
        self._antialias_running = False
        self._antialias_cancel = None
        self._sync_antialias_button_enabled()
        self._finish_shared_progress_if_idle()
        self._refresh_image_list_item_states()
        if summary.cancelled:
            self._set_progress_status("antialias_stopped_status")
            self._append_log(
                self._tr(
                    "antialias_stopped_log",
                    "Сглаживание векторов остановлено."
                    if self._ui_language == "ru"
                    else "Vector antialiasing stopped.",
                )
            )
            return
        self._set_progress_status("antialias_finished_status")
        if summary.failed:
            self._append_log(
                self._tr(
                    "antialias_opened_failed_log",
                    "Сглаживание CIF: сохранено {saved}/{changed}, ошибки: {errors}"
                    if self._ui_language == "ru"
                    else "CIF antialiasing: saved {saved}/{changed}, errors: {errors}",
                    saved=summary.saved_count,
                    changed=summary.changed_count,
                    errors="; ".join(summary.failed[:4]),
                )
            )
            return
        self._append_log(
            self._tr(
                "antialias_opened_done_log",
                "Сглаживание применено и сохранено для {count} CIF, grade={grade}."
                if self._ui_language == "ru"
                else "Antialiasing applied and saved for {count} CIF files, grade={grade}.",
                count=summary.saved_count,
                grade=summary.grade,
            )
        )
