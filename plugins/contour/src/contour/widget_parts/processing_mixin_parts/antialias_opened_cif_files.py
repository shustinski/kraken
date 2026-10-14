"""processing_mixin: antialias opened cif files."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    AntialiasCifRunnable,
    AntialiasCifWorkItem,
    Path,
    PolygonData,
    threading,
)


class AntialiasOpenedCifFilesMixin:
    def _antialias_opened_cif_files(self: Any) -> None:
        if hasattr(self, "polygon_editor") and self.polygon_editor.vector_edits_locked():
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
        if not self._confirm_processing_question(
            self._tr(
                "antialias_all_vectors_confirm",
                "Запустить сглаживание всех векторов?"
                if self._ui_language == "ru"
                else "Start antialiasing of all vectors?",
            )
        ):
            return
        cif_entries = list(self._collect_opened_cif_entries().items())
        if not cif_entries:
            self._append_log(
                self._tr(
                    "antialias_all_vectors_none_log",
                    "В проекте нет CIF для сглаживания."
                    if self._ui_language == "ru"
                    else "The project has no CIF files to antialias.",
                )
            )
            return
        grade = int(self.antialias_grade_spin.value()) if hasattr(self, "antialias_grade_spin") else 1
        current_path = self._workspace.current_image_path
        if current_path is not None:
            self._workspace.update_current_polygons(self.get_polygons())
        cached_by_path = {
            str(Path(image_path)): state for image_path, state in self._workspace.cached_states()
        }
        work_items: list[AntialiasCifWorkItem] = []
        for stem, cif_path in cif_entries:
            image_path = self._image_path_for_cif_stem(stem)
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
                AntialiasCifWorkItem(
                    stem=stem,
                    cif_path=str(cif_path),
                    image_path=image_path,
                    polygons=snapshot_polygons,
                    image_size=image_size,
                )
            )
        self._antialias_run_id = int(getattr(self, "_antialias_run_id", 0)) + 1
        cancel_event = threading.Event()
        self._antialias_cancel = cancel_event
        self._antialias_running = True
        self._sync_antialias_button_enabled()
        self._show_batch_progress(len(work_items))
        self._set_progress_status("antialias_started_status")
        self._antialias_thread_pool.start(
            AntialiasCifRunnable(
                items=tuple(work_items),
                grade=grade,
                signals=self._antialias_signals,
                cancel_event=cancel_event,
                run_id=self._antialias_run_id,
            )
        )
