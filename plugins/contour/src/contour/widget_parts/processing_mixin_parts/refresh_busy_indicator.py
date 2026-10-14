"""processing_mixin: refresh busy indicator."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    AutoTuneResult,
)


class RefreshBusyIndicatorMixin:
    def _append_log(self: Any, message: str) -> None:
        self.logMessage.emit(message)

    def _refresh_busy_indicator(self: Any) -> None:
        active = any(
            (
                self._preview_running_request_id is not None,
                self._preview_pending_request is not None,
                self._preview_update_timer.isActive(),
                self._prepared_image_running_request_id is not None,
                self._prepared_image_pending_request is not None,
                self._auto_tune_running_request_id is not None,
            )
        )
        if hasattr(self, "preview_busy_label"):
            self.preview_busy_label.setText(self._busy_indicator_text())
            self.preview_busy_label.setToolTip(f"{self._busy_indicator_text()} - {self._busy_progress_value}%")
            self.preview_busy_label.setVisible(active)
        if hasattr(self, "preview_busy_progress"):
            if active:
                self.preview_busy_progress.setValue(self._busy_progress_value)
            self.preview_busy_progress.setVisible(active)
        if active and self._preview_running_request_id is not None:
            if not self._busy_progress_timer.isActive():
                self._busy_progress_timer.start()
        else:
            self._busy_progress_timer.stop()

    def _on_prepared_image_result(
        self: Any, request_id: int, image_path: str, preprocessed_image, pipeline_config: dict
    ) -> None:
        if request_id != self._prepared_image_running_request_id:
            return
        if pipeline_config != self.get_pipeline():
            return
        if self._workspace.store_preprocessed_image(image_path, preprocessed_image, pipeline_config):
            self._refresh_current_display_image_only(preserve_view=True)
            self._try_extract_if_recognition_enabled()

    def _on_prepared_image_error(self: Any, request_id: int, message: str) -> None:
        if request_id != self._prepared_image_running_request_id:
            return
        self._append_log(self._tr("processing_failed_log", error=message))

    def _on_prepared_image_finished(self: Any, request_id: int) -> None:
        if request_id == self._prepared_image_running_request_id:
            self._prepared_image_running_request_id = None
            self._prepared_image_running_signature = None
            self._prepared_image_run_cancel = None
        if self._prepared_image_pending_request is not None:
            self._start_pending_prepared_image_update()
        elif self._preview_pending_request is not None and not self._preview_update_timer.isActive():
            self._start_pending_preview_processing()
        self._refresh_busy_indicator()

    def _on_auto_tune_result(self: Any, request_id: int, result: AutoTuneResult) -> None:
        if request_id != self._auto_tune_running_request_id:
            return
        self._apply_auto_tune_result(result)
        roi_width = result.roi_bbox[2]
        roi_height = result.roi_bbox[3]
        self._append_log(
            self._tr(
                "auto_tune_finished_log",
                "Автоподбор завершён: score={score:.3f}, ROI={width}x{height}, проверок={evaluations}."
                if self._ui_language == "ru"
                else "Auto-fit completed: score={score:.3f}, ROI={width}x{height}, evaluations={evaluations}.",
                score=result.score,
                width=roi_width,
                height=roi_height,
                evaluations=result.evaluations,
            )
        )

    def _on_auto_tune_error(self: Any, request_id: int, message: str) -> None:
        if request_id != self._auto_tune_running_request_id:
            return
        self._append_log(
            self._tr(
                "auto_tune_failed_log",
                "Ошибка автоподбора: {error}" if self._ui_language == "ru" else "Auto-fit failed: {error}",
                error=message,
            )
        )

    def _on_auto_tune_finished(self: Any, request_id: int) -> None:
        if request_id == self._auto_tune_running_request_id:
            self._auto_tune_running_request_id = None
        self._refresh_busy_indicator()
