"""presenter: ui context state."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from PyQt6.QtCore import QSignalBlocker
from ...core.analysis_modes import (
    confidence_metric_family,
    geometry_mode_for_object_type,
)
from ...core.domain import BuildOptions, BuildResult
from ...core.repository import (
    metric_value_for_record,
)
from ...ui.ui_constants import (
    DEFAULT_MATRIX_LAYOUT_MODE,
    DEFAULT_MATRIX_SCORE_VIEW_MODE,
    DEFAULT_MATRIX_METRIC_KEY,
    DEFAULT_POINT_EXTRACTION_MODE,
    DEFAULT_POLYGON_CONFIDENCE_SUMMARY,
    MATRIX_METRIC_OPTIONS,
)
from neuralimage.Validation_gradient_widget_lite.app.state import ExtendMatrixTabState


class UiContextStateMixin:
    def _set_ui_context_from_state(self, state: ExtendMatrixTabState) -> None:
        analysis_blocker = QSignalBlocker(self.analysis_mode_combo)
        analysis_index = self.analysis_mode_combo.findData(str(state.analysis_mode))
        self.analysis_mode_combo.setCurrentIndex(analysis_index if analysis_index >= 0 else 0)
        del analysis_blocker

        geometry_value = geometry_mode_for_object_type(state.object_type).value
        geometry_blocker = QSignalBlocker(self.geometry_mode_combo)
        geometry_index = self.geometry_mode_combo.findData(str(geometry_value))
        self.geometry_mode_combo.setCurrentIndex(geometry_index if geometry_index >= 0 else 0)
        del geometry_blocker

        score_view_blocker = QSignalBlocker(self.matrix_score_view_combo)
        score_view_index = self.matrix_score_view_combo.findData(str(state.matrix_score_view_mode or DEFAULT_MATRIX_SCORE_VIEW_MODE))
        self.matrix_score_view_combo.setCurrentIndex(score_view_index if score_view_index >= 0 else 0)
        del score_view_blocker

        layout_blocker = QSignalBlocker(self.layout_mode_combo)
        layout_index = self.layout_mode_combo.findData(str(state.layout_config.mode or DEFAULT_MATRIX_LAYOUT_MODE))
        self.layout_mode_combo.setCurrentIndex(layout_index if layout_index >= 0 else 0)
        del layout_blocker

        total_frames_blocker = QSignalBlocker(self.total_frames_spin)
        self.total_frames_spin.setValue(int(state.layout_config.total_frames))
        del total_frames_blocker

        frames_per_row_blocker = QSignalBlocker(self.frames_per_row_spin)
        self.frames_per_row_spin.setValue(int(state.layout_config.frames_per_row))
        del frames_per_row_blocker

        rows_blocker = QSignalBlocker(self.matrix_rows_spin)
        self.matrix_rows_spin.setValue(int(state.layout_config.rows))
        del rows_blocker

        columns_blocker = QSignalBlocker(self.matrix_columns_spin)
        self.matrix_columns_spin.setValue(int(state.layout_config.columns))
        del columns_blocker

        frame_type_filter_blocker = QSignalBlocker(self.frame_type_filter_combo)
        frame_type_filter_index = self.frame_type_filter_combo.findData(str(state.frame_type_filter or 'all'))
        self.frame_type_filter_combo.setCurrentIndex(frame_type_filter_index if frame_type_filter_index >= 0 else 0)
        del frame_type_filter_blocker

    def _analysis_options_from_controls(self, state: ExtendMatrixTabState, *, object_type: str) -> BuildOptions:
        return replace(
            state.build_result.options,
            geometry_mode=geometry_mode_for_object_type(object_type),
            tile_mode=self._selected_tile_mode(),
            tile_width=int(self.tile_width_spin.value()),
            tile_height=int(self.tile_height_spin.value()),
            tile_overlap_mode=self._selected_tile_overlap_mode(),
            tile_overlap=int(self.tile_overlap_spin.value()),
            subpixel_view_mode=self._selected_subpixel_view_mode(),
            subpixel_rows=self._selected_subpixel_rows(),
            subpixel_columns=self._selected_subpixel_columns(),
            subpixel_aggregation=self._selected_subpixel_aggregation(),
            mask_threshold=float(self.mask_threshold_spin.value()),
            boundary_radius=int(self.boundary_radius_spin.value()),
            confidence_uncertainty_delta=self._selected_confidence_uncertainty_delta(),
            point_match_radius=float(self.point_match_radius_spin.value()),
            point_confidence_radius=int(self.point_confidence_radius_spin.value()),
            point_extraction_mode=str(self.point_extraction_mode_combo.currentData() or DEFAULT_POINT_EXTRACTION_MODE),
            polygon_confidence_summary=str(self.polygon_confidence_summary_combo.currentData() or DEFAULT_POLYGON_CONFIDENCE_SUMMARY),
        )

    def _invalidate_state_runtime_caches(self, state: ExtendMatrixTabState, *, clear_metric_results: bool = False) -> None:
        state.percentile_cache.clear()
        state.base_records_cache.clear()
        state.repeated_percentile_cache.clear()
        if clear_metric_results:
            state.metric_result_cache.clear()

    def _begin_worker_request(self, *, state: ExtendMatrixTabState | None) -> int:
        self._request_generation += 1
        self._active_request_generation = self._request_generation
        self._active_processing_keys = set()
        self._active_progress_current = 0
        self._active_progress_total = 0
        self._active_progress_key = ""
        if state is not None:
            state.processing_state_by_key.clear()
            state.matrix_view.set_processing_keys(set())
        return self._request_generation

    def _is_active_request_generation(self, generation: int | None) -> bool:
        return generation is None or generation == self._active_request_generation

    def _analytics_request_signature(self, state: ExtendMatrixTabState, metric_key: str | None = None) -> tuple[object, ...]:
        return (
            "analytics",
            id(state.widget),
            str(state.analysis_mode),
            str(state.object_type),
            str(metric_key or state.metric_key or DEFAULT_MATRIX_METRIC_KEY),
            str(state.confidence_model_id or ""),
            state.build_result.options,
        )

    def _update_processing_visuals(self, state: ExtendMatrixTabState | None) -> None:
        if state is None:
            return
        state.matrix_view.set_processing_keys(set(self._active_processing_keys))

    def _progress_format_text(self, current: int, total: int, key: str) -> str:
        frame_label = Path(key).name if key else ""
        running_count = len(self._active_processing_keys)
        parts: list[str] = []
        if total > 0:
            parts.append(f"{current}/{total}")
        else:
            parts.append("Working...")
        if running_count > 0:
            parts.append(f"active {running_count}")
        if frame_label:
            parts.append(frame_label)
        return " | ".join(parts)

    def _on_frame_state_changed(self, key: str, status: str, *, generation: int | None = None) -> None:
        if not self._is_active_request_generation(generation):
            return
        state = self._active_compute_state
        if state is None or not key:
            return
        normalized_key = str(key)
        normalized_status = str(status or "running").lower()
        state.processing_state_by_key[normalized_key] = normalized_status
        if normalized_status == "running":
            self._active_processing_keys.add(normalized_key)
        else:
            self._active_processing_keys.discard(normalized_key)
        self._update_processing_visuals(state)
        if self.build_progress.isVisible():
            self.build_progress.setFormat(self._progress_format_text(self._active_progress_current, self._active_progress_total, self._active_progress_key))

    def _sync_current_analysis_context(self, state: ExtendMatrixTabState, *, auto_recompute: bool) -> None:
        selected_analysis_mode = self._selected_analysis_mode()
        selected_object_type = self._selected_object_type()
        updated_options = self._analysis_options_from_controls(state, object_type=selected_object_type)
        options_changed = updated_options != state.build_result.options
        if options_changed:
            state.build_result = replace(state.build_result, options=updated_options)
            self._invalidate_state_runtime_caches(state, clear_metric_results=True)

        state.analysis_mode = selected_analysis_mode
        state.object_type = selected_object_type
        state.frame_type_filter = str(self.frame_type_filter_combo.currentData() or state.frame_type_filter or 'all')
        state.build_result = replace(state.build_result, options=updated_options)
        selected_confidence_model_id = self._selected_confidence_model_id(state.build_result)
        state.confidence_model_id = str(selected_confidence_model_id or "") or None
        state.metric_scope = str(state.confidence_model_id or "")

        self._sync_metric_controls(
            state.build_result,
            preferred_metric_key=state.metric_key,
            preferred_scope_key=state.confidence_model_id or state.metric_scope,
            context_state=state,
        )
        metric_key = str(self.metric_combo.currentData() or self._default_metric_key_for_state(state, state.build_result))
        state.metric_key = metric_key

        if self._worker_thread is not None:
            return
        requires_analytics = options_changed or self._metric_value_missing_for_build_result(state.build_result, metric_key)
        if requires_analytics:
            if auto_recompute and bool(getattr(state.build_result, "scores_computed", False)):
                self._start_compute_analytics(state=state, sync_context=False)
            return
        self._apply_metric_to_state(state, metric_key)

    def _metric_group_for_key(self, metric_key: str) -> str:
        for _label, key, group in MATRIX_METRIC_OPTIONS:
            if key == metric_key:
                return str(group)
        return "overall"

    def _build_result_has_labeled(self, build_result: BuildResult | None) -> bool:
        return bool(build_result is not None and any(record.gt_path for record in build_result.records))

    def _metric_scope_for_metric_key(self, metric_key: str) -> str:
        family = confidence_metric_family(metric_key)
        if family is None:
            return ""
        _metric_family, model_id = family
        return str(model_id)

    def _metric_key_has_values(self, build_result: BuildResult | None, metric_key: str) -> bool:
        if build_result is None:
            return False
        for record in build_result.records:
            if metric_value_for_record(record, metric_key) is not None:
                return True
        return False
