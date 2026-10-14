"""processing_mixin: queue preview processing."""
from __future__ import annotations
from typing import Any
from ...infrastructure.filter_application_profiler import FilterApplicationProfile
from ...infrastructure.profiling import (
    filter_application_profiling_enabled,
    write_profile_report,
)
from contour.widget_parts._imports import (
    VIA_SEARCH_MODE_TEMPLATE,
    Path,
    PolygonData,
    PreparedImageRequest,
    PreparedImageRunnable,
    PreviewProcessingRequest,
    build_prepared_image_signature,
    build_preview_request_signature,
    normalize_via_search_mode,
    threading,
)


class QueuePreviewProcessingMixin:
    def _on_neighbor_frame_activated(self: Any, image_path: str) -> None:
        if image_path in self._workspace.image_paths:
            if self._image_path_in_image_list(image_path):
                self._set_image_list_current_path(image_path, fallback_to_first=False)
            self._workspace._current_image_path = str(Path(image_path))
            self._workspace._current_state = None
            self.load_image(image_path)

    def _abort_in_flight_interactive_processing(self: Any, *, preview: bool, prepared: bool) -> None:
        self._geometry_validation_generation = int(getattr(self, "_geometry_validation_generation", 0)) + 1
        self._preview_update_timer.stop()
        if preview:
            if self._preview_run_cancel is not None:
                self._preview_run_cancel.set()
            self._preview_pending_request = None
            self._preview_pending_signature = None
            self._preview_pending_save_result = False
            self._preview_running_save_result = False
        if prepared:
            if self._prepared_image_run_cancel is not None:
                self._prepared_image_run_cancel.set()
            self._prepared_image_pending_request = None
            self._prepared_image_pending_signature = None
        self._refresh_busy_indicator()

    def _queue_prepared_image_update(self: Any, image_path: str, source_image) -> None:
        request = PreparedImageRequest(
            image_path=image_path,
            source_image=source_image,
            pipeline_config=self.get_pipeline(),
        )
        signature = self._prepared_image_request_signature(request)
        if signature == self._prepared_image_running_signature or signature == self._prepared_image_pending_signature:
            self._refresh_busy_indicator()
            return
        if self._prepared_image_run_cancel is not None:
            self._prepared_image_run_cancel.set()
        self._prepared_image_pending_request = request
        self._prepared_image_pending_signature = signature
        self._refresh_busy_indicator()
        self._start_pending_prepared_image_update()

    def _start_pending_prepared_image_update(self: Any) -> None:
        if self._prepared_image_pending_request is None:
            return
        if self._preview_running_request_id is not None:
            if self._preview_run_cancel is not None:
                self._preview_run_cancel.set()
            return
        if self._prepared_image_running_request_id is not None:
            if self._prepared_image_run_cancel is not None:
                self._prepared_image_run_cancel.set()
            return
        request = self._prepared_image_pending_request
        self._prepared_image_pending_request = None
        request_signature = self._prepared_image_pending_signature
        self._prepared_image_pending_signature = None
        self._prepared_image_request_serial += 1
        request_id = self._prepared_image_request_serial
        self._prepared_image_running_request_id = request_id
        self._prepared_image_running_signature = request_signature
        cancel = threading.Event()
        self._prepared_image_run_cancel = cancel
        worker = PreparedImageRunnable(
            request_id=request_id,
            request=request,
            cancel_event=cancel,
            profile=filter_application_profiling_enabled(),
        )
        worker.signals.profile.connect(self._on_filter_application_profile)
        worker.signals.result.connect(self._on_prepared_image_result)
        worker.signals.error.connect(self._on_prepared_image_error)
        worker.signals.finished.connect(self._on_prepared_image_finished)
        self._prepared_image_thread_pool.start(worker)
        self._refresh_busy_indicator()

    def _on_filter_application_profile(
        self: Any,
        request_id: int,
        profile: FilterApplicationProfile,
        status: str,
    ) -> None:
        if request_id != self._prepared_image_running_request_id:
            status = "superseded"
        write_profile_report(profile.format_summary(status=status), profile.format_stats())

    def _build_preview_request(self: Any) -> PreviewProcessingRequest | None:
        if not self._workspace.current_image_path:
            return None
        source_image = None
        preprocessed_image = None
        current_state = self._workspace.current_state
        pipeline_config = self.get_pipeline()
        if current_state is not None and current_state.image_path == self._workspace.current_image_path:
            source_image = current_state.source_image
            if current_state.preprocessed_image is not None and current_state.pipeline_config == pipeline_config:
                preprocessed_image = current_state.preprocessed_image
        passthrough: tuple[PolygonData, ...] | None = None
        if hasattr(self, "recognition_mode_combo") and str(self.recognition_mode_combo.currentData() or "") == "disabled":
            passthrough = tuple(polygon.clone() for polygon in self.get_polygons())
        return PreviewProcessingRequest(
            image_path=self._workspace.current_image_path,
            pipeline_config=pipeline_config,
            contour_settings=self._current_contour_settings(),
            source_image=source_image,
            preprocessed_image=preprocessed_image,
            passthrough_polygons=passthrough,
        )

    def _preview_request_signature(self: Any, request: PreviewProcessingRequest) -> tuple[str, str, str, int]:
        return build_preview_request_signature(request)

    def _prepared_image_request_signature(self: Any, request: PreparedImageRequest) -> tuple[str, str]:
        return build_prepared_image_signature(request)

    def _queue_preview_processing(self: Any, *, debounced: bool, save_result: bool = False) -> None:
        request = self._build_preview_request()
        if request is None:
            self._append_log(self._tr("no_image_selected_log"))
            return
        if (
            normalize_via_search_mode(request.contour_settings.via_search_mode) == VIA_SEARCH_MODE_TEMPLATE
            and not list(getattr(request.contour_settings, "via_template_images", []) or [])
        ):
            self._append_log("Для режима поиска по шаблону добавьте хотя бы один шаблон")
        if hasattr(self, "recognition_mode_combo"):
            self._set_recognition_status("updating")
        signature = self._preview_request_signature(request)
        contact_session = self._contact_profile_waiting_for_preview()
        force_contact_rerun = bool(
            contact_session is not None and contact_session.waiting_for_preview
        )
        if signature == self._preview_running_signature and not force_contact_rerun:
            self._preview_running_save_result = self._preview_running_save_result or save_result
            self._refresh_busy_indicator()
            return
        if signature == self._preview_pending_signature:
            self._preview_pending_save_result = self._preview_pending_save_result or save_result
            self._refresh_busy_indicator()
            return
        self._start_image_recognition_profile(request)
        self._preview_update_timer.stop()
        if self._preview_run_cancel is not None:
            self._preview_run_cancel.set()
            self._preview_running_save_result = False
        self._preview_pending_request = request
        self._preview_pending_signature = signature
        self._preview_pending_save_result = bool(save_result)
        self._refresh_busy_indicator()
        if debounced:
            self._preview_update_timer.start()
            return
        self._preview_update_timer.stop()
        self._start_pending_preview_processing()
