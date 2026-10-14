"""processing_mixin: fix internal contours all."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    FixInternalContoursCifWorkItem,
    Path,
    PolygonData,
    QMessageBox,
    threading,
)

from contour.widget_parts.processing_mixin_parts.module_state import _LOGGER


class FixInternalContoursAllMixin:
    def _fix_internal_contours_all_cifs(self: Any) -> None:
        if hasattr(self, "polygon_editor") and self.polygon_editor.vector_edits_locked():
            ru = self._ui_language == "ru"
            QMessageBox.information(
                self.window(),
                self._tr(
                    "fix_internal_contours_stats_title",
                    "Исправление внутренних контуров" if ru else "Fix internal contours",
                ),
                self._tr(
                    "fix_internal_contours_locked_log",
                    "Сначала завершите текущее редактирование векторов."
                    if ru
                    else "Finish the current vector edit session first.",
                ),
            )
            return
        if self._batch_controller.is_running:
            self._append_log(self._tr("batch_already_running_log"))
            return
        if getattr(self, "_antialias_running", False):
            self._append_log(
                self._tr(
                    "antialias_already_running_log",
                    "Сглаживание векторов уже выполняется."
                    if self._ui_language == "ru"
                    else "Vector antialiasing is already running.",
                )
            )
            return
        if getattr(self, "_fix_internal_contours_running", False):
            self._append_log(
                self._tr(
                    "fix_internal_contours_already_running_log",
                    "Исправление внутренних контуров уже выполняется."
                    if self._ui_language == "ru"
                    else "Fix internal contours is already running.",
                )
            )
            return
        if not self._confirm_processing_question(
            self._tr(
                "fix_internal_contours_confirm",
                "Запустить исправление внутренних контуров?"
                if self._ui_language == "ru"
                else "Run internal contour correction?",
            )
        ):
            return
        cif_entries = list(self._collect_opened_cif_entries().items())
        if not cif_entries:
            ru = self._ui_language == "ru"
            message = self._tr(
                "fix_internal_contours_none_log",
                "В проекте нет CIF для проверки." if ru else "The project has no CIF files to check.",
            )
            self._append_log(message)
            QMessageBox.information(
                self.window(),
                self._tr(
                    "fix_internal_contours_stats_title",
                    "Исправление внутренних контуров" if ru else "Fix internal contours",
                ),
                message,
            )
            return
        try:
            current_path = self._workspace.current_image_path
            if current_path is not None:
                self._workspace.update_current_polygons(self.get_polygons())
            cached_by_path = {
                str(Path(image_path)): state for image_path, state in self._workspace.cached_states()
            }
            work_items: list[FixInternalContoursCifWorkItem] = []
            for stem, cif_path in cif_entries:
                image_path = self._image_path_for_cif_stem(stem)
                if image_path is None:
                    cif_file = Path(str(cif_path))
                    image_path = self._image_path_for_cif_stem(cif_file.stem)
                image_key = str(Path(image_path)) if image_path else None
                state = cached_by_path.get(image_key) if image_key else None
                snapshot_polygons: tuple[PolygonData, ...] | None = None
                image_size: tuple[int, int] | None = None
                if state is not None and state.polygons:
                    snapshot_polygons = tuple(polygon.clone() for polygon in state.polygons)
                    if state.source_image is not None:
                        image_size = (
                            int(state.source_image.shape[1]),
                            int(state.source_image.shape[0]),
                        )
                work_items.append(
                    FixInternalContoursCifWorkItem(
                        stem=stem,
                        cif_path=str(cif_path),
                        image_path=image_path,
                        polygons=snapshot_polygons,
                        image_size=image_size,
                    )
                )
            self._fix_internal_contours_run_id = int(getattr(self, "_fix_internal_contours_run_id", 0)) + 1
            cancel_event = threading.Event()
            self._fix_internal_contours_cancel = cancel_event
            self._fix_internal_contours_running = True
            run_dock = getattr(self.window(), "_run_dock", None)
            if run_dock is not None:
                run_dock.show()
            self._show_batch_progress(len(work_items))
            self._set_progress_status("fix_internal_contours_started_status")
            self._begin_fix_internal_contours_batch(
                tuple(work_items),
                run_id=self._fix_internal_contours_run_id,
                cancel_event=cancel_event,
            )
        except Exception as exc:
            _LOGGER.exception("Failed to start internal-contour repair")
            self._fix_internal_contours_running = False
            self._fix_internal_contours_cancel = None
            self._finish_shared_progress_if_idle()
            ru = self._ui_language == "ru"
            QMessageBox.critical(
                self.window(),
                self._tr(
                    "fix_internal_contours_stats_title",
                    "Исправление внутренних контуров" if ru else "Fix internal contours",
                ),
                str(exc),
            )
