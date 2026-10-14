"""presenter: start compute analytics."""
from __future__ import annotations
from datetime import datetime
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QMessageBox
from ...core.analysis_modes import (
    default_confidence_model_id,
)
from ...core.domain import BuildResult
from ...core.workers import AnalyticsWorker
from ...ui.ui_constants import (
    DEFAULT_MATRIX_METRIC_KEY,
)
from neuralimage.Validation_gradient_widget_lite.app.state import ExtendMatrixTabState


class StartComputeAnalyticsMixin:
    def _start_compute_analytics(
        self,
        *,
        state: ExtendMatrixTabState | None = None,
        sync_context: bool = True,
        apply_pending_controls: bool = False,
    ) -> None:
        state = state or self._current_tab_state()
        if state is None:
            return
        if sync_context:
            self._sync_current_analysis_context(state, auto_recompute=False)
        if apply_pending_controls:
            self._apply_pending_display_controls(state)
        metric_key = str(self.metric_combo.currentData() or state.metric_key or DEFAULT_MATRIX_METRIC_KEY)
        request_signature = self._analytics_request_signature(state, metric_key)
        self._worker_kind = "analytics"
        self._active_compute_state = state
        self._worker_thread = QThread(self._view)
        self._worker = AnalyticsWorker(state.build_result, metric_key)
        generation = self._begin_worker_request(state=state)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda current, total, key, g=generation: self._on_build_progress(current, total, key, generation=g))
        if hasattr(self._worker, "frameStateChanged"):
            self._worker.frameStateChanged.connect(lambda key, status, g=generation: self._on_frame_state_changed(key, status, generation=g))
        self._worker.finished.connect(lambda result, g=generation, s=request_signature: self._on_analytics_finished(result, generation=g, request_signature=s))
        self._worker.failed.connect(lambda message, g=generation: self._on_worker_failed(message, generation=g))
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)
        self._worker_thread.start()
        self._show_progress_bar(visible=True, format_text="Computing analytics...")
        self._sync_action_buttons()

    def _on_compute_requested(self) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        previous_options = state.build_result.options
        self._sync_current_analysis_context(state, auto_recompute=False)
        self._apply_pending_display_controls(state)
        metric_key = str(self.metric_combo.currentData() or state.metric_key or DEFAULT_MATRIX_METRIC_KEY)
        options_changed = state.build_result.options != previous_options
        needs_analytics = (
            not bool(getattr(state.build_result, "scores_computed", False))
            or options_changed
            or self._metric_value_missing_for_build_result(state.build_result, metric_key)
        )
        if not needs_analytics:
            self._apply_metric_to_state(state, metric_key)
            self._sync_action_buttons()
            return
        self._start_compute_analytics(state=state, sync_context=False, apply_pending_controls=False)

    def _request_cancel_build(self) -> None:
        if self._worker is None:
            return
        request_cancel = getattr(self._worker, "request_cancel", None)
        if callable(request_cancel):
            request_cancel()
        self._show_progress_bar(visible=True, format_text="Cancelling...")

    def _cleanup_worker(self) -> None:
        active_state = self._active_compute_state
        if self._worker is not None:
            self._worker.deleteLater()
        if self._worker_thread is not None:
            self._worker_thread.deleteLater()
        if active_state is not None:
            active_state.matrix_view.set_processing_keys(set())
            active_state.processing_state_by_key.clear()
        self._worker = None
        self._worker_thread = None
        self._worker_kind = None
        self._active_compute_state = None
        self._active_request_generation = None
        self._active_processing_keys = set()
        self._active_progress_current = 0
        self._active_progress_total = 0
        self._active_progress_key = ""
        self._pending_build_snapshot = None
        self._sync_action_buttons()
        deferred_restart = self._deferred_analytics_restart
        self._deferred_analytics_restart = None
        if deferred_restart is not None:
            restart_state, sync_context = deferred_restart
            if restart_state.widget in self._tab_states:
                self._start_compute_analytics(state=restart_state, sync_context=sync_context)

    def _on_build_progress(self, current: int, total: int, key: str, *, generation: int | None = None) -> None:
        if not self._is_active_request_generation(generation):
            return
        self._active_progress_current = int(current)
        self._active_progress_total = int(total)
        self._active_progress_key = str(key or "")
        if self._active_compute_state is not None and not self._active_processing_keys:
            self._active_compute_state.matrix_view.set_processing_keys({str(key)} if key else set())
        format_text = self._progress_format_text(current, total, str(key or ""))
        self._show_progress_bar(visible=True, current=current, total=total, key=key, format_text=format_text)

    def _on_build_finished(self, result: BuildResult, *, generation: int | None = None) -> None:
        if not self._is_active_request_generation(generation):
            return
        snapshot = self._pending_build_snapshot or self._capture_view_snapshot()
        snapshot["confidence_model_id"] = snapshot.get("confidence_model_id") or snapshot.get("metric_scope") or default_confidence_model_id(result)
        self._sync_metric_controls(
            result,
            preferred_metric_key=str(snapshot.get("metric_key") or DEFAULT_MATRIX_METRIC_KEY),
            preferred_scope_key=str(snapshot.get("confidence_model_id") or ""),
        )
        snapshot["metric_scope"] = str(self.metric_scope_combo.currentData() or "")
        snapshot["confidence_model_id"] = str(self.metric_scope_combo.currentData() or "")
        snapshot["metric_key"] = str(self.metric_combo.currentData() or self._default_metric_key_for_state(None, result))
        state = self._create_matrix_tab(result, snapshot)
        self._connect_histogram_cards(state)
        ok = self._apply_tab_visual_settings(state, reset_view=True, update_histograms=False)
        self._show_progress_bar(visible=False)
        if not ok:
            state.widget.deleteLater()
            return
        title = f"{len(result.model_specs)} models [{datetime.now().strftime('%H:%M:%S')}]"
        self._tab_states[state.widget] = state
        tab_index = self.matrix_tabs.addTab(state.widget, title)
        self.matrix_tabs.setCurrentIndex(tab_index)
        if result.records:
            state.matrix_view.select_record_by_key(result.records[0].key, ensure_visible=False)
            self._update_matrix_preview(state, result.records[0])
        self._schedule_metric_histogram_update(state)
        self._sync_action_buttons()

    def _on_analytics_finished(self, result: BuildResult, *, generation: int | None = None, request_signature: tuple[object, ...] | None = None) -> None:
        if not self._is_active_request_generation(generation):
            return
        state = self._active_compute_state or self._current_tab_state()
        self._show_progress_bar(visible=False)
        if state is None:
            return
        if request_signature is not None and request_signature != self._analytics_request_signature(state):
            self._deferred_analytics_restart = (state, False)
            return
        state.matrix_view.set_processing_keys(set())
        state.processing_state_by_key.clear()
        state.build_result = result
        self._invalidate_state_runtime_caches(state, clear_metric_results=True)
        self._sync_metric_controls(
            result,
            preferred_metric_key=result.selected_metric_key,
            preferred_scope_key=state.confidence_model_id or state.metric_scope,
            context_state=state,
        )
        state.confidence_model_id = self._selected_confidence_model_id(result)
        state.metric_scope = str(state.confidence_model_id or "")
        state.metric_key = str(self.metric_combo.currentData() or result.selected_metric_key or self._default_metric_key_for_state(state, result))
        self._apply_metric_to_state(state, state.metric_key)
        self._sync_action_buttons()

    def _on_worker_failed(self, message: str, *, generation: int | None = None) -> None:
        if not self._is_active_request_generation(generation):
            return
        self._show_progress_bar(visible=False)
        if self._active_compute_state is not None:
            self._active_compute_state.matrix_view.set_processing_keys(set())
            self._active_compute_state.processing_state_by_key.clear()
        if message and "cancel" in message.lower():
            self._sync_action_buttons()
            return
        QMessageBox.critical(self._view, self._t("dialog.warning_title"), message or self._t("errors.background_failed"))
