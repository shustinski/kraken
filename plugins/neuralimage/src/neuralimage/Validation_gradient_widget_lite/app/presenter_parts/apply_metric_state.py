"""presenter: apply metric state."""
from __future__ import annotations
from dataclasses import replace
from math import isfinite
from PyQt6.QtCore import QSignalBlocker
from ...core.domain import FrameRecord
from ...core.repository import (
    compute_metric_percentiles,
    metric_value_for_record,
)
from ...ui.details_dialog import ExtendFrameDetailsDialog
from ...ui.ui_constants import (
    DEFAULT_MATRIX_METRIC_KEY,
)
from neuralimage.Validation_gradient_widget_lite.app.state import ExtendMatrixTabState


class ApplyMetricStateMixin:
    def _apply_metric_to_state(self, state: ExtendMatrixTabState, metric_key: str) -> None:
        available = set(state.build_result.available_metric_keys or ())
        if available and metric_key not in available:
            metric_key = self._default_metric_key_for_state(state, state.build_result)
        state.metric_key = metric_key
        self._invalidate_state_runtime_caches(state, clear_metric_results=False)
        self.metric_combo.setToolTip(self._metric_hint_fallback(metric_key, state.build_result))
        cached_build_result = state.metric_result_cache.get(metric_key)
        if cached_build_result is not None:
            state.build_result = cached_build_result
            self._apply_tab_visual_settings(state, reset_view=False)
            return
        if self._metric_value_missing_for_build_result(state.build_result, metric_key):
            if self._worker is None and bool(getattr(state.build_result, "scores_computed", False)):
                self._start_compute_analytics(state=state, sync_context=False)
            else:
                self._apply_tab_visual_settings(state, reset_view=False)
            return
        updated_records: list[FrameRecord] = []
        absolute_scores: list[float] = []
        higher_is_better = self._metric_higher_is_better(metric_key)
        for record in state.build_result.records:
            value = metric_value_for_record(record, metric_key)
            numeric = float(value) if value is not None and isfinite(float(value)) else 0.0
            absolute_scores.append(numeric)
        min_absolute = min(absolute_scores) if absolute_scores else 0.0
        max_absolute = max(absolute_scores) if absolute_scores else 0.0
        span = max(1e-8, max_absolute - min_absolute)
        for record, absolute in zip(state.build_result.records, absolute_scores):
            relative = 0.0 if abs(max_absolute - min_absolute) <= 1e-8 else (absolute - min_absolute) / span
            display = relative if higher_is_better else (1.0 - relative)
            updated_records.append(replace(record, score=float(display), absolute_score=float(absolute), relative_score=float(relative), score_ready=True))
        percentile_map = compute_metric_percentiles(updated_records, metric_key)
        updated_records = [replace(record, score_percentile=float(percentile_map.get(record.key, 0.0))) for record in updated_records]
        state.build_result = replace(state.build_result, records=tuple(updated_records), min_score=min((record.score for record in updated_records), default=0.0), max_score=max((record.score for record in updated_records), default=0.0), min_absolute_score=min_absolute, max_absolute_score=max_absolute, selected_metric_key=metric_key)
        state.metric_result_cache[metric_key] = state.build_result
        self._apply_tab_visual_settings(state, reset_view=False)

    def _on_metric_group_changed(self, *_args) -> None:
        state = self._current_tab_state()
        build_result = None if state is None else state.build_result
        preferred_group = str(self.metric_group_combo.currentData() or "overall")
        self._sync_metric_controls(build_result, preferred_group_key=preferred_group, context_state=state)
        if state is None:
            return
        self._apply_metric_to_state(state, str(self.metric_combo.currentData() or DEFAULT_MATRIX_METRIC_KEY))

    def _on_metric_scope_changed(self, *_args) -> None:
        state = self._current_tab_state()
        preferred_scope = str(self.metric_scope_combo.currentData() or "")
        build_result = None if state is None else state.build_result
        self._sync_metric_controls(build_result, preferred_scope_key=preferred_scope, context_state=state)
        if state is None:
            return
        self.metric_combo.setToolTip(self._metric_hint_fallback(str(self.metric_combo.currentData() or DEFAULT_MATRIX_METRIC_KEY), state.build_result))
        self._sync_action_buttons()

    def _on_analysis_mode_changed(self, *_args) -> None:
        state = self._current_tab_state()
        self._sync_mode_controls(state, None if state is None else state.build_result)
        self._sync_action_buttons()

    def _on_object_type_changed(self, *_args) -> None:
        state = self._current_tab_state()
        self._sync_mode_controls(state, None if state is None else state.build_result)
        self._sync_action_buttons()

    def _on_polygon_compare_profile_changed(self, *_args) -> None:
        self._apply_polygon_compare_profile(self._selected_polygon_compare_profile())
        self._sync_action_buttons()

    def _on_subpixel_view_mode_changed(self, *_args) -> None:
        state = self._current_tab_state()
        self._sync_tile_overlap_bounds()
        self._sync_mode_controls(state, None if state is None else state.build_result)
        self._update_subpixel_plan_label(None if state is None else state.build_result)
        self._sync_action_buttons()

    def _on_subpixel_grid_parameter_changed(self, *_args) -> None:
        state = self._current_tab_state()
        self._sync_tile_overlap_bounds()
        self._sync_mode_controls(state, None if state is None else state.build_result)
        self._update_subpixel_plan_label(None if state is None else state.build_result)
        self._sync_action_buttons()

    def _on_tile_mode_changed(self, *_args) -> None:  # pragma: no cover - compatibility shim
        self._on_subpixel_view_mode_changed()

    def _on_tile_overlap_mode_changed(self, *_args) -> None:  # pragma: no cover - compatibility shim
        state = self._current_tab_state()
        self._sync_action_buttons()

    def _on_tile_grid_parameter_changed(self, *_args) -> None:
        self._sync_tile_overlap_bounds()
        self._on_subpixel_grid_parameter_changed()

    def _on_metric_changed(self, *_args) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        metric_key = str(self.metric_combo.currentData() or DEFAULT_MATRIX_METRIC_KEY)
        self.metric_combo.setToolTip(self._metric_hint_fallback(metric_key, state.build_result))
        self._sync_action_buttons()

    def _on_frame_type_filter_changed(self, *_args) -> None:
        self._sync_action_buttons()

    def _on_matrix_score_view_changed(self, *_args) -> None:
        self._sync_action_buttons()

    def _on_matrix_visual_parameter_changed(self, *_args) -> None:
        state = self._current_tab_state()
        self._sync_mode_controls(state, None if state is None else state.build_result)
        self._sync_action_buttons()

    def _on_current_tab_changed(self, _index: int) -> None:
        state = self._current_tab_state()
        if state is None:
            self._sync_action_buttons()
            return
        self._set_ui_context_from_state(state)
        scope_blocker = QSignalBlocker(self.metric_scope_combo)
        self._populate_metric_scope_combo(state.build_result, state.confidence_model_id or state.metric_scope)
        scope_index = self.metric_scope_combo.findData(str(state.confidence_model_id or state.metric_scope or ""))
        self.metric_scope_combo.setCurrentIndex(scope_index if scope_index >= 0 else 0)
        del scope_blocker
        self._sync_metric_controls(
            state.build_result,
            preferred_metric_key=state.metric_key,
            preferred_scope_key=state.confidence_model_id or state.metric_scope,
            context_state=state,
        )
        state.metric_key = str(self.metric_combo.currentData() or state.metric_key or self._default_metric_key_for_state(state, state.build_result))
        self.metric_combo.setToolTip(self._metric_hint_fallback(state.metric_key, state.build_result))
        if self._metric_value_missing_for_build_result(state.build_result, state.metric_key):
            self._update_matrix_preview(state)
            if self._worker is None and bool(getattr(state.build_result, "scores_computed", False)):
                self._start_compute_analytics(state=state, sync_context=False)
            elif self._worker is not None and bool(getattr(state.build_result, "scores_computed", False)):
                self._deferred_analytics_restart = (state, False)
            self._sync_action_buttons()
            return
        if not bool(getattr(state.build_result, "scores_computed", False)) or str(getattr(state.build_result, "selected_metric_key", "")) != state.metric_key:
            self._apply_metric_to_state(state, state.metric_key)
            self._sync_action_buttons()
            return
        self._update_matrix_preview(state)
        self._sync_action_buttons()

    def _close_matrix_tab(self, index: int) -> None:
        widget = self.matrix_tabs.widget(index)
        if widget is None:
            return
        self._tab_states.pop(widget, None)
        self.matrix_tabs.removeTab(index)
        widget.deleteLater()
        self._sync_action_buttons()

    def _on_matrix_overview_changed(self, state: ExtendMatrixTabState, image, visible_rect, selected_position, selected_blink_on, processing_positions, reference_position) -> None:
        state.mini_map.set_overview(image, visible_rect, selected_position, selected_blink_on, processing_positions, reference_position)

    def _on_record_selected(self, state: ExtendMatrixTabState, record: FrameRecord | None) -> None:
        if self._current_tab_state() is state:
            self._update_matrix_preview(state, record)
            self._sync_action_buttons()

    def _on_tile_selected(self, state: ExtendMatrixTabState, selection: object | None) -> None:
        if self._current_tab_state() is state:
            record = getattr(selection, "record", None)
            self._update_matrix_preview(state, record if isinstance(record, FrameRecord) else None)
            self._sync_action_buttons()

    def _forget_details_dialog(self, dialog: ExtendFrameDetailsDialog) -> None:
        self._details_dialogs = [opened for opened in self._details_dialogs if opened is not dialog]

    def _close_all_details_dialogs(self) -> None:
        for dialog in list(self._details_dialogs):
            try:
                dialog.close()
            except Exception:
                pass
        self._details_dialogs.clear()
