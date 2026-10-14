"""extraction_settings_mixin: start auto tune reference."""
from __future__ import annotations
from contour.widget_parts._imports import (
    AutoTuneRunnable,
)


class StartAutoTuneReferenceMixin:
    def _start_auto_tune_from_reference(self) -> None:
        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        reference_polygons = self.get_polygons()

        if current_state is None or current_state.source_image is None or current_image_path is None:
            self._append_log(
                self._tr(
                    "no_image_selected_log",
                    "Изображение не выбрано." if self._ui_language == "ru" else "No image selected.",
                )
            )
            return
        if not reference_polygons:
            self._append_log(
                self._tr(
                    "auto_tune_no_reference_log",
                    "Для автоподбора сначала нарисуйте эталонный полигон или область."
                    if self._ui_language == "ru"
                    else "Draw at least one reference polygon before running auto-fit.",
                )
            )
            return
        if self._auto_tune_running_request_id is not None:
            self._append_log(
                self._tr(
                    "auto_tune_already_running_log",
                    "Автоподбор уже выполняется." if self._ui_language == "ru" else "Auto-fit is already running.",
                )
            )
            return

        self._auto_tune_request_serial += 1
        request_id = self._auto_tune_request_serial
        self._auto_tune_running_request_id = request_id
        self._append_log(
            self._tr(
                "auto_tune_started_log",
                "Запущен автоподбор по {count} полигонам."
                if self._ui_language == "ru"
                else "Auto-fit started using {count} reference polygons.",
                count=len(reference_polygons),
            )
        )
        worker = AutoTuneRunnable(
            request_id=request_id,
            image_path=current_image_path,
            source_image=current_state.source_image,
            reference_polygons=reference_polygons,
        )
        worker.signals.result.connect(self._on_auto_tune_result)
        worker.signals.error.connect(self._on_auto_tune_error)
        worker.signals.finished.connect(self._on_auto_tune_finished)
        self._auto_tune_thread_pool.start(worker)
        self._refresh_busy_indicator()
