"""processing_mixin: fix internal contours finished."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    FixInternalContoursCifItemResult,
    FixInternalContoursCifJobSummary,
    Path,
    QMessageBox,
)


class FixInternalContoursFinishedMixin:
    def _on_fix_internal_contours_progress(self: Any, run_id: int, current: int, total: int) -> None:
        if run_id != getattr(self, "_fix_internal_contours_run_id", 0):
            return
        if self._batch_progress_enabled:
            self._apply_batch_progress_bar(current, total)
        self._set_progress_status("batch_progress_status", current=current, total=total)

    def _on_fix_internal_contours_item_finished(self: Any, result: FixInternalContoursCifItemResult) -> None:
        if getattr(self, "_closing", False):
            return
        if int(result.run_id) != getattr(self, "_fix_internal_contours_run_id", 0):
            return
        if result.error or not result.changed:
            return
        image_path = result.image_path
        image_key = str(Path(image_path)) if image_path else None
        current_path = self._workspace.current_image_path
        current_key = str(Path(current_path)) if current_path else None
        fixed_polygons = [polygon.clone() for polygon in result.polygons]
        if image_key:
            for cached_path, state in self._workspace.cached_states():
                if str(Path(cached_path)) != image_key:
                    continue
                state.polygons = [polygon.clone() for polygon in fixed_polygons]
                state.reference_polygons = [polygon.clone() for polygon in fixed_polygons]
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
                self.polygon_editor.set_polygons(fixed_polygons)
            finally:
                self._updating_views = False
            self._update_vector_edit_status_label()

    def _on_fix_internal_contours_finished(self: Any, summary: FixInternalContoursCifJobSummary) -> None:
        if int(summary.run_id) != getattr(self, "_fix_internal_contours_run_id", 0):
            return
        self._fix_internal_contours_running = False
        self._fix_internal_contours_cancel = None
        self._finish_shared_progress_if_idle()
        self._refresh_image_list_item_states()
        stats = summary.stats
        ru = self._ui_language == "ru"
        if stats.cancelled:
            self._set_progress_status("fix_internal_contours_stopped_status")
            self._append_log(
                self._tr(
                    "fix_internal_contours_stopped_log",
                    "Исправление внутренних контуров остановлено."
                    if ru
                    else "Fix internal contours stopped.",
                )
            )
        else:
            self._set_progress_status("fix_internal_contours_finished_status")
            if stats.failed:
                self._append_log(
                    self._tr(
                        "fix_internal_contours_failed_log",
                        "Исправление внутренних контуров: сохранено {saved}, ошибки: {errors}"
                        if ru
                        else "Fix internal contours: saved {saved}, errors: {errors}",
                        saved=summary.saved_count,
                        errors="; ".join(stats.failed[:4]),
                    )
                )
            else:
                self._append_log(
                    self._tr(
                        "fix_internal_contours_done_log",
                        "Исправление внутренних контуров: проверено CIF {checked}, исправлено {fixed}, без изменений {unchanged}, пропущено keyhole KLayout {skipped}."
                        if ru
                        else "Fix internal contours: checked {checked} CIF files, fixed {fixed}, unchanged {unchanged}, skipped KLayout keyholes {skipped}.",
                        checked=stats.checked_cif_files,
                        fixed=stats.fixed_cif_files,
                        unchanged=stats.unchanged_cif_files,
                        skipped=stats.skipped_klayout_keyholes,
                    )
                )
        stats_lines = [
            (
                f"Проверено CIF: {stats.checked_cif_files}"
                if ru
                else f"CIF files checked: {stats.checked_cif_files}"
            ),
            (
                f"Проверено семейств: {stats.checked_families}"
                if ru
                else f"Families checked: {stats.checked_families}"
            ),
            (
                f"Исправлено CIF: {stats.fixed_cif_files}"
                if ru
                else f"CIF files fixed: {stats.fixed_cif_files}"
            ),
            (
                f"Исправлено семейств: {stats.fixed_families}"
                if ru
                else f"Families fixed: {stats.fixed_families}"
            ),
            (
                f"Исправлено областей отверстий: {stats.fixed_hole_regions}"
                if ru
                else f"Hole regions fixed: {stats.fixed_hole_regions}"
            ),
            (
                f"Без изменений: {stats.unchanged_cif_files}"
                if ru
                else f"Unchanged: {stats.unchanged_cif_files}"
            ),
            (
                f"Пропущено KLayout keyhole: {stats.skipped_klayout_keyholes}"
                if ru
                else f"Skipped KLayout keyholes: {stats.skipped_klayout_keyholes}"
            ),
        ]
        if stats.failed:
            stats_lines.append(
                f"Ошибки: {len(stats.failed)}" if ru else f"Errors: {len(stats.failed)}"
            )
        QMessageBox.information(
            self.window(),
            self._tr(
                "fix_internal_contours_stats_title",
                "Исправление внутренних контуров" if ru else "Fix internal contours",
            ),
            "\n".join(stats_lines),
        )
