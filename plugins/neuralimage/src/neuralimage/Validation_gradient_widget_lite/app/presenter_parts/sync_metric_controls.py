"""presenter: sync metric controls."""
from __future__ import annotations
from dataclasses import replace
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox
from ...core.analysis_modes import (
    INTRA_MODEL_CONFIDENCE_MODE,
    default_confidence_model_id,
    default_metric_key,
    percentile_basis_keys,
    resolve_analysis_context,
)
from ...core.backend_constants import (
    BCE_SCORE_CAP,
)
from ...core.domain import BuildResult, FrameIdentity, FrameRecord
from ...core.repository import (
    _parse_model_metric_key,
    metric_higher_is_better,
)
from ...core.subpixel_grid import subpixel_spec_from_options
from ...ui.matrix_view import MatrixLayoutConfig, build_matrix_layout
from ...ui.ui_constants import (
    DEFAULT_GRADIENT_NAME,
    DEFAULT_MATRIX_SCORE_VIEW_MODE,
    DEFAULT_MATRIX_METRIC_KEY,
    DEFAULT_SUBPIXEL_AGGREGATION,
    MATRIX_METRIC_OPTIONS,
)
from neuralimage.Validation_gradient_widget_lite.app.state import ExtendMatrixTabState


class SyncMetricControlsMixin:
    def _visible_metric_keys(self, build_result: BuildResult | None, metric_scope: str | None) -> tuple[str, ...]:
        state = self._current_tab_state()
        if state is None:
            return tuple()
        return self._display_metric_keys_for_state(state, build_result or state.build_result)

    def _available_metric_groups(self, build_result: BuildResult | None) -> list[tuple[str, str]]:
        groups: list[tuple[str, str]] = [("Overall frame", "overall")]
        if build_result is not None and len(build_result.model_specs) >= 2:
            groups.append(("Model vs model", "model_model"))
        if self._build_result_has_labeled(build_result):
            groups.append(("Model vs labeled frames", "model_labeled"))
        return groups

    def _metric_items_for_group(self, group_key: str, build_result: BuildResult | None) -> list[tuple[str, str]]:
        available_keys = set(build_result.available_metric_keys if build_result is not None else ())
        items: list[tuple[str, str]] = []
        for label, key, group in MATRIX_METRIC_OPTIONS:
            if str(group) != str(group_key):
                continue
            if available_keys and key not in available_keys:
                continue
            items.append((str(label), str(key)))
        return items

    def _sync_metric_controls(
        self,
        build_result: BuildResult | None,
        preferred_metric_key: str | None = None,
        preferred_group_key: str | None = None,
        preferred_scope_key: str | None = None,
        context_state: ExtendMatrixTabState | None = None,
    ) -> None:
        state = context_state if context_state is not None else self._current_tab_state()
        self._sync_mode_controls(state, build_result)
        metric_key = str(preferred_metric_key or self.metric_combo.currentData() or DEFAULT_MATRIX_METRIC_KEY)
        selected_confidence_model_id = str(preferred_scope_key or self.metric_scope_combo.currentData() or self._metric_scope_for_metric_key(metric_key) or "")
        self._populate_metric_scope_combo(build_result, selected_confidence_model_id)
        context = self._analysis_context_for_state(state, build_result)
        if build_result is not None and context.analysis_mode == INTRA_MODEL_CONFIDENCE_MODE:
            selected_confidence_model_id = str(self.metric_scope_combo.currentData() or default_confidence_model_id(build_result) or "")
            context = resolve_analysis_context(
                build_result,
                context.analysis_mode,
                context.object_type,
                confidence_model_id=selected_confidence_model_id,
            )
        basis_keys = [str(key) for key in percentile_basis_keys(context) if build_result is None or key in set(build_result.available_metric_keys)]
        if build_result is not None and not basis_keys:
            basis_keys = self._fallback_metric_keys_for_build_result(build_result)
        if metric_key not in basis_keys:
            metric_key = basis_keys[0] if basis_keys else default_metric_key(context)

        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        for key in basis_keys:
            label = self._metric_label(str(key), build_result)
            self.metric_combo.addItem(label, key)
            combo_index = self.metric_combo.count() - 1
            self.metric_combo.setItemData(combo_index, self._metric_hint_fallback(key, build_result), Qt.ItemDataRole.ToolTipRole)
        metric_index = self.metric_combo.findData(metric_key)
        self.metric_combo.setCurrentIndex(metric_index if metric_index >= 0 else 0)
        self.metric_combo.setToolTip(self._metric_hint_fallback(metric_key, build_result))
        self.metric_combo.blockSignals(False)

    def _attach_matrix_coordinates(self, records: tuple[FrameRecord, ...] | list[FrameRecord], layout_config: MatrixLayoutConfig) -> tuple[FrameRecord, ...]:
        records_tuple = tuple(records)
        placements, _columns, _rows = build_matrix_layout(list(records_tuple), layout_config)
        records_by_key: dict[str, FrameRecord] = {}
        changed = False
        for placement_index, (record, row, column) in enumerate(placements):
            identity = record.identity
            if identity is None:
                identity = FrameIdentity(frame_id=placement_index, base_id=placement_index, tile_x=column, tile_y=row, source_key=record.key)
                records_by_key[record.key] = replace(record, identity=identity)
                changed = True
                continue
            if int(identity.tile_x if identity.tile_x is not None else -1) == int(column) and int(identity.tile_y if identity.tile_y is not None else -1) == int(row):
                records_by_key[record.key] = record
                continue
            else:
                identity = replace(identity, tile_x=column, tile_y=row)
            records_by_key[record.key] = replace(record, identity=identity)
            changed = True
        if not changed:
            return records_tuple
        return tuple(records_by_key.get(record.key, record) for record in records_tuple)

    def _sync_state_record_coordinates(self, state: ExtendMatrixTabState) -> None:
        attached_records = self._attach_matrix_coordinates(state.build_result.records, state.layout_config)
        if attached_records is not state.build_result.records:
            state.build_result = replace(state.build_result, records=tuple(attached_records))

    def _apply_pending_display_controls(self, state: ExtendMatrixTabState) -> None:
        state.layout_config = self._build_layout_config()
        state.matrix_score_view_mode = str(self.matrix_score_view_combo.currentData() or DEFAULT_MATRIX_SCORE_VIEW_MODE)
        state.frame_type_filter = str(self.frame_type_filter_combo.currentData() or 'all')

    def _apply_tab_visual_settings(self, state: ExtendMatrixTabState, *, reset_view: bool = False, update_histograms: bool = True) -> bool:
        try:
            self._sync_state_record_coordinates(state)
            display_records = self._display_records_for_state(state)
            self._update_subpixel_plan_label(state.build_result)
            state.matrix_view.set_gradient_preset(DEFAULT_GRADIENT_NAME)
            state.matrix_view.set_cell_size(int(state.cell_size))
            state.matrix_view.set_layout_config(state.layout_config)
            state.matrix_view.set_subpixel_comparison_mode(getattr(state.build_result.options, "comparison_mode", None))
            options = state.build_result.options
            if str(getattr(options, "subpixel_view_mode", "pixel") or "pixel") == "tile":
                subpixel_spec = subpixel_spec_from_options(options, self._tile_source_shape_for_result(state.build_result))
            else:
                subpixel_spec = None
            state.matrix_view.set_subpixel_grid_spec(
                subpixel_spec,
                aggregation=str(getattr(options, "subpixel_aggregation", DEFAULT_SUBPIXEL_AGGREGATION) or DEFAULT_SUBPIXEL_AGGREGATION),
            )
            state.matrix_view.set_score_view_mode(str(state.matrix_score_view_mode or DEFAULT_MATRIX_SCORE_VIEW_MODE))
            state.matrix_view.set_metric_context(
                state.metric_key,
                point_match_radius=float(getattr(state.build_result.options, "point_match_radius", 3.0)),
                bce_score_cap=float(BCE_SCORE_CAP),
            )
            state.matrix_view.set_reference_key(state.build_result.best_match_key)
            sort_mode = "input_order" if str(state.layout_config.mode or "indexed_grid") == "manual_grid" else "name"
            state.matrix_view.set_records(list(display_records), sort_mode=sort_mode, reset_view=reset_view)
            self._update_matrix_preview(state)
            if update_histograms:
                self._update_metric_histograms(state)
        except ValueError as error:
            QMessageBox.warning(self._view, self._t("errors.layout"), str(error))
            return False
        return True

    def _metric_value_missing_for_build_result(self, build_result: BuildResult | None, metric_key: str) -> bool:
        if build_result is None:
            return False
        parsed = _parse_model_metric_key(metric_key)
        if parsed is None:
            return False
        family, _model_id = parsed
        if family == 'model_uncertain_fraction':
            frame_type = next((str(record.summary.frame_type) for record in build_result.records if record.summary is not None and getattr(record.summary, 'frame_type', None)), None)
            if frame_type == 'point':
                return False
        if family == 'model_point_contrast':
            frame_type = next((str(record.summary.frame_type) for record in build_result.records if record.summary is not None and getattr(record.summary, 'frame_type', None)), None)
            if frame_type == 'polygon':
                return False
        for record in build_result.records:
            summary = record.summary
            if summary is None:
                continue
            if metric_key in getattr(summary, 'metric_values', {}):
                return False
        return True

    def _metric_higher_is_better(self, metric_key: str) -> bool:
        return metric_higher_is_better(metric_key)
