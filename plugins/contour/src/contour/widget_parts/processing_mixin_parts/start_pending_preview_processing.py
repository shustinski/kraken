"""processing_mixin: start pending preview processing."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    PreviewProcessingRunnable,
    threading,
)


class StartPendingPreviewProcessingMixin:
    def _start_pending_preview_processing(self: Any) -> None:
        if self._preview_pending_request is None:
            return
        if (
            self._prepared_image_running_request_id is not None
            or self._prepared_image_pending_request is not None
        ):
            if self._prepared_image_running_request_id is None:
                self._start_pending_prepared_image_update()
            return
        if self._preview_running_request_id is not None:
            if self._preview_run_cancel is not None:
                self._preview_run_cancel.set()
            return
        request = self._preview_pending_request
        self._preview_pending_request = None
        request_signature = self._preview_pending_signature
        self._preview_pending_signature = None
        save_result = self._preview_pending_save_result
        self._preview_pending_save_result = False
        self._preview_request_serial += 1
        request_id = self._preview_request_serial
        self._preview_running_request_id = request_id
        self._preview_running_signature = request_signature
        self._preview_running_save_result = save_result
        cancel = threading.Event()
        self._preview_run_cancel = cancel
        self._reset_busy_progress(request)
        self._busy_progress_timer.start()
        contact_session = self._contact_profile_waiting_for_preview()
        profile_contact = bool(
            contact_session is not None and contact_session.waiting_for_preview
        )
        recognition_session = getattr(self, "_image_recognition_profile", None)
        profile_recognition = bool(recognition_session is not None)
        if profile_contact:
            contact_session.stop()
        if recognition_session is not None:
            recognition_session.stop()
        worker = PreviewProcessingRunnable(
            request_id=request_id,
            request=request,
            cancel_event=cancel,
            profile=profile_contact or profile_recognition,
        )
        if profile_contact:
            contact_session.preview_request_id = request_id
            contact_session.note("preview_started")
            worker.signals.profile.connect(self._on_contact_preview_profile)
        if recognition_session is not None:
            recognition_session.preview_request_id = request_id
            recognition_session.note("worker_started")
            if not profile_contact:
                worker.signals.profile.connect(self._on_contact_preview_profile)
        worker.signals.result.connect(self._on_preview_processing_result)
        worker.signals.error.connect(self._on_preview_processing_error)
        worker.signals.finished.connect(self._on_preview_processing_finished)
        self._preview_thread_pool.start(worker)
        self._refresh_busy_indicator()
