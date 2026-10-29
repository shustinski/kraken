"""processing_mixin: preview processing result."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
    QTimer,
)


class PreviewProcessingResultMixin:
    def _on_preview_processing_result(self: Any, request_id: int, result) -> None:
        if request_id != self._preview_running_request_id:
            return
        is_current = self._workspace.current_image_path == result.image_path
        if is_current:
            self._resume_preview_main_profiles(request_id)
        recognition_session = getattr(self, "_image_recognition_profile", None)
        if (
            is_current
            and recognition_session is not None
            and recognition_session.preview_request_id == request_id
        ):
            recognition_session.note("result_received")

        if (
            is_current
            and hasattr(self, "recognition_mode_combo")
            and str(self.recognition_mode_combo.currentData() or "") == "via"
        ):
            result.polygons = self._recognized_contact_polygons_with_diameter(
                list(result.polygons),
                diameter=int(self.via_output_diameter_spin.value()),
            )
        self._remember_produced_recognition(result.image_path, list(result.polygons))
        applied_to_current = self._workspace.apply_processing_result(result)
        if not is_current or not applied_to_current:
            return
        self._sync_current_state_views(preserve_view=True, sync_neighbors=False)
        self._update_frame_item_status(result.image_path)
        if hasattr(self, "recognition_mode_combo"):
            if str(self.recognition_mode_combo.currentData() or "") == "disabled":
                self._set_recognition_status("disabled")
            else:
                self._set_recognition_status("idle")
        self._set_progress_status("current_image_processed_status")
        self._append_log(
            self._tr(
                "current_image_processed_log",
                image_name=Path(result.image_path).name,
                count=len(result.polygons),
            )
        )
        if self._preview_running_save_result:
            self.save_current_result(polygons=list(result.polygons))
        self.imageProcessed.emit(result.image_path, result.polygons)
        if (
            recognition_session is not None
            and recognition_session.preview_request_id == request_id
        ):
            recognition_session.polygon_count = len(result.polygons)
            recognition_session.note("result_applied_to_ui")
            QTimer.singleShot(
                0,
                lambda: self._finish_image_recognition_profile("displayed"),
            )
        for attribute, finish in (
            ("_contact_deletion_profile", self._finish_contact_deletion_profile),
            ("_contact_placement_profile", self._finish_contact_placement_profile),
        ):
            contact_session = getattr(self, attribute, None)
            if contact_session is None or contact_session.preview_request_id != request_id:
                continue
            contact_session.note("result_applied_to_ui")
            QTimer.singleShot(0, lambda callback=finish: callback("displayed"))
            break

    def _on_preview_processing_error(self: Any, request_id: int, message: str) -> None:
        if request_id != self._preview_running_request_id:
            return
        self._resume_preview_main_profiles(request_id)
        if hasattr(self, "recognition_mode_combo"):
            self._set_recognition_status("error", message)
        self._append_log(self._tr("processing_failed_log", error=message))
        recognition_session = getattr(self, "_image_recognition_profile", None)
        if (
            recognition_session is not None
            and recognition_session.preview_request_id == request_id
        ):
            recognition_session.note("processing_error")
            self._finish_image_recognition_profile("processing_error")
        for attribute, finish in (
            ("_contact_deletion_profile", self._finish_contact_deletion_profile),
            ("_contact_placement_profile", self._finish_contact_placement_profile),
        ):
            contact_session = getattr(self, attribute, None)
            if contact_session is not None and contact_session.preview_request_id == request_id:
                finish("processing_error")
                break

    def _on_preview_processing_finished(self: Any, request_id: int) -> None:
        cancelled_finish = None
        for attribute, finish in (
            ("_contact_deletion_profile", self._finish_contact_deletion_profile),
            ("_contact_placement_profile", self._finish_contact_placement_profile),
        ):
            contact_session = getattr(self, attribute, None)
            if (
                contact_session is not None
                and contact_session.preview_request_id == request_id
                and "result_applied_to_ui" not in contact_session.timings_ms
            ):
                cancelled_finish = finish
                break
        if request_id == self._preview_running_request_id:
            self._preview_running_request_id = None
            self._preview_running_signature = None
            self._preview_running_save_result = False
            self._preview_run_cancel = None
            self._preview_running_request_for_progress = None
            self._busy_progress_stage = ""
            self._busy_progress_value = 0
        if self._prepared_image_pending_request is not None:
            self._start_pending_prepared_image_update()
        elif self._preview_pending_request is not None and not self._preview_update_timer.isActive():
            self._start_pending_preview_processing()
        self._refresh_busy_indicator()
        if cancelled_finish is not None:
            cancelled_finish("cancelled")
        recognition_session = getattr(self, "_image_recognition_profile", None)
        if (
            recognition_session is not None
            and recognition_session.preview_request_id == request_id
            and "result_applied_to_ui" not in recognition_session.timings_ms
            and "processing_error" not in recognition_session.timings_ms
        ):
            self._finish_image_recognition_profile("cancelled")
