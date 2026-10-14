"""presenter: sync mode controls."""
from __future__ import annotations
import re
from math import isfinite
from pathlib import Path
from PyQt6.QtCore import QThread, Qt
from ...core.analysis_modes import (
    INTRA_MODEL_CONFIDENCE_MODE,
    INTER_MODEL_ANALYSIS_MODE,
    POINT_OBJECT_TYPE,
    default_metric_key,
    display_metric_keys,
    object_type_from_geometry_mode,
    percentile_basis_keys,
    resolve_analysis_context,
)
from ...core.domain import BuildResult, FolderSpec, ModelSpec
from ...infra.services import ValidationGradientLiteSettingsService
from ...ui.details_dialog import ExtendFrameDetailsDialog
from ...ui.ui_constants import (
    DEFAULT_CONFIDENCE_UNCERTAINTY_DELTA,
    DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE,
    DEFAULT_MATRIX_LAYOUT_MODE,
    DEFAULT_POLYGON_COMPARE_PROFILE,
    FOLDER_CHECKED_ROLE,
    FOLDER_CONFIDENCE_ROLE,
    FOLDER_LABEL_ROLE,
    CONFIDENCE_UNCERTAINTY_PROFILE_VALUES,
    POLYGON_COMPARE_PROFILE_VALUES,
)
from neuralimage.Validation_gradient_widget_lite.app.state import ExtendMatrixTabState


class SyncModeControlsMixin:
    def __init__(self, view, settings_service: ValidationGradientLiteSettingsService) -> None:
        super().__init__(view)
        self._view = view
        self._settings_service = settings_service
        self._worker_thread: QThread | None = None
        self._worker = None
        self._worker_kind: str | None = None
        self._active_compute_state: ExtendMatrixTabState | None = None
        self._original_folder: FolderSpec | None = None
        self._gt_folder: FolderSpec | None = None
        self._folder_check_guard = False
        self._tab_states: dict[object, ExtendMatrixTabState] = {}
        self._pending_build_snapshot: dict[str, object] | None = None
        self._details_dialogs: list[ExtendFrameDetailsDialog] = []
        self._details_view_payload: dict[str, object] = self._settings_service.load_details_view_payload() or {}
        self._request_generation = 0
        self._active_request_generation: int | None = None
        self._active_processing_keys: set[str] = set()
        self._active_progress_current = 0
        self._active_progress_total = 0
        self._active_progress_key = ""
        self._deferred_analytics_restart: tuple[ExtendMatrixTabState, bool] | None = None
        self._histogram_update_generation = 0

    def __getattr__(self, name: str):
        return getattr(self._view, name)

    def _current_tab_state(self) -> ExtendMatrixTabState | None:
        widget = self.matrix_tabs.currentWidget()
        if widget is None:
            return None
        return self._tab_states.get(widget)

    @staticmethod
    def _set_row_visible(row: object | None, visible: bool) -> None:
        if row is not None and hasattr(row, "setVisible"):
            row.setVisible(bool(visible))

    def _selected_analysis_mode(self) -> str:
        return str(self.analysis_mode_combo.currentData() or INTER_MODEL_ANALYSIS_MODE)

    def _selected_object_type(self) -> str:
        return object_type_from_geometry_mode(self.geometry_mode_combo.currentData())

    def _selected_confidence_model_id(self, build_result: BuildResult | None) -> str | None:
        selected = self.metric_scope_combo.currentData()
        current = str(selected) if selected is not None else None
        return resolve_analysis_context(
            build_result,
            self._selected_analysis_mode(),
            self._selected_object_type(),
            confidence_model_id=current,
        ).confidence_model_id

    def _analysis_context_for_state(self, state: ExtendMatrixTabState | None, build_result: BuildResult | None = None):
        active_build_result = build_result if build_result is not None else (state.build_result if state is not None else None)
        analysis_mode = state.analysis_mode if state is not None else self._selected_analysis_mode()
        object_type = state.object_type if state is not None else self._selected_object_type()
        confidence_model_id = state.confidence_model_id if state is not None else self._selected_confidence_model_id(active_build_result)
        return resolve_analysis_context(
            active_build_result,
            analysis_mode,
            object_type,
            confidence_model_id=confidence_model_id,
        )

    def _display_metric_keys_for_state(self, state: ExtendMatrixTabState | None, build_result: BuildResult | None = None) -> tuple[str, ...]:
        context = self._analysis_context_for_state(state, build_result)
        return tuple(key for key in display_metric_keys(context) if build_result is None or key in set(build_result.available_metric_keys))

    def _percentile_basis_keys_for_state(self, state: ExtendMatrixTabState | None, build_result: BuildResult | None = None) -> tuple[str, ...]:
        context = self._analysis_context_for_state(state, build_result)
        return tuple(key for key in percentile_basis_keys(context) if build_result is None or key in set(build_result.available_metric_keys))

    def _default_metric_key_for_state(self, state: ExtendMatrixTabState | None, build_result: BuildResult | None = None) -> str:
        context = self._analysis_context_for_state(state, build_result)
        key = default_metric_key(context)
        available = set((build_result or (state.build_result if state is not None else None)).available_metric_keys) if (build_result or state) is not None and getattr((build_result or (state.build_result if state is not None else None)), "available_metric_keys", None) is not None else set()
        if not available or key in available:
            return key
        for candidate in self._percentile_basis_keys_for_state(state, build_result):
            if candidate in available:
                return candidate
        for candidate in ("overall_frame_score", "export_priority_score", "model_model_score", "disagreement_score"):
            if candidate in available:
                return candidate
        return next(iter(sorted(available)), "overall_frame_score")

    def _fallback_metric_keys_for_build_result(self, build_result: BuildResult | None) -> list[str]:
        available = set(build_result.available_metric_keys if build_result is not None else ())
        candidates = ("overall_frame_score", "export_priority_score", "model_model_score", "disagreement_score")
        keys = [key for key in candidates if not available or key in available]
        return keys or [next(iter(sorted(available)), "overall_frame_score")]

    def _confidence_context_available(self, context, build_result: BuildResult | None) -> bool:
        return context.analysis_mode == INTRA_MODEL_CONFIDENCE_MODE and context.confidence_model_id is not None

    def _sync_mode_controls(self, state: ExtendMatrixTabState | None = None, build_result: BuildResult | None = None) -> None:
        context = self._analysis_context_for_state(state, build_result)
        is_confidence = self._confidence_context_available(context, build_result)
        is_point = context.object_type == POINT_OBJECT_TYPE
        tile_mode_enabled = self._selected_subpixel_view_mode() == "tile"
        self._set_row_visible(getattr(self, "_matrix_pixel_size_row", None), False)
        self._set_row_enabled(getattr(self, "_matrix_layout_row", None), True)
        layout_mode = str(self.layout_mode_combo.currentData() or DEFAULT_MATRIX_LAYOUT_MODE)
        is_indexed_layout = layout_mode == "indexed_grid"
        self._set_row_enabled(getattr(self, "_matrix_total_frames_row", None), is_indexed_layout)
        self._set_row_enabled(getattr(self, "_matrix_frames_per_row_row", None), is_indexed_layout)
        self._set_row_enabled(getattr(self, "_matrix_rows_row", None), not is_indexed_layout)
        self._set_row_enabled(getattr(self, "_matrix_columns_row", None), not is_indexed_layout)
        self._set_row_visible(getattr(self, "_metric_scope_row", None), is_confidence)
        self._set_row_visible(getattr(self, "_metric_select_row", None), not is_confidence)
        self._set_row_visible(getattr(self, "_matrix_confidence_delta_row", None), is_confidence)
        self._set_row_visible(getattr(self, "_matrix_polygon_confidence_summary_row", None), is_confidence and not is_point)
        self._set_row_visible(getattr(self, "_matrix_polygon_compare_profile_row", None), not is_confidence and not is_point)
        self._set_row_visible(getattr(self, "_matrix_point_radius_row", None), not is_confidence and is_point)
        self._set_row_visible(getattr(self, "_matrix_point_confidence_radius_row", None), is_confidence and is_point)
        self._set_row_visible(getattr(self, "_matrix_point_mode_row", None), is_point)
        self._set_row_visible(getattr(self, "_matrix_frame_type_filter_row", None), False)
        self._set_row_visible(getattr(self, "_subpixel_rows_row", None), False)
        self._set_row_visible(getattr(self, "_subpixel_columns_row", None), False)
        self._set_row_visible(getattr(self, "_tile_width_row", None), tile_mode_enabled)
        self._set_row_visible(getattr(self, "_tile_height_row", None), tile_mode_enabled)
        self._set_row_visible(getattr(self, "_tile_overlap_row", None), tile_mode_enabled)
        self._set_row_enabled(getattr(self, "_subpixel_aggregation_row", None), tile_mode_enabled)
        self._set_row_enabled(getattr(self, "_subpixel_plan_row", None), True)
        self._update_subpixel_plan_label(state.build_result if state is not None else build_result)

    def _checked_model_specs(self) -> tuple[ModelSpec, ...]:
        specs: list[ModelSpec] = []
        threshold, _boundary_radius = self._selected_polygon_compare_values()
        for row in range(self.folder_list.count()):
            item = self.folder_list.item(row)
            if not bool(item.data(FOLDER_CHECKED_ROLE)):
                continue
            folder_path = Path(item.data(Qt.ItemDataRole.UserRole))
            label = str(item.data(FOLDER_LABEL_ROLE) or folder_path.name)
            confidence_path_text = str(item.data(FOLDER_CONFIDENCE_ROLE) or "").strip()
            confidence_folder = Path(confidence_path_text) if confidence_path_text else None
            model_id = re.sub(r"[^a-zA-Z0-9_]+", "_", label.strip().lower()).strip("_") or f"model_{row + 1}"
            specs.append(ModelSpec(
                model_id=model_id,
                display_name=label,
                mask_folder=folder_path,
                prob_folder=confidence_folder,
                threshold=threshold,
            ))
        return tuple(specs)

    def _selected_confidence_uncertainty_profile(self) -> str:
        return str(self.confidence_uncertainty_profile_combo.currentData() or DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE)

    def _confidence_uncertainty_delta_for_profile(self, profile_key: str | None) -> float:
        profile = str(profile_key or DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE)
        value = CONFIDENCE_UNCERTAINTY_PROFILE_VALUES.get(profile, DEFAULT_CONFIDENCE_UNCERTAINTY_DELTA)
        return float(value)

    def _selected_confidence_uncertainty_delta(self) -> float:
        return self._confidence_uncertainty_delta_for_profile(self._selected_confidence_uncertainty_profile())

    def _confidence_uncertainty_profile_for_value(self, value: float | None) -> str:
        if value is None or not isfinite(float(value)):
            return DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE
        numeric = float(value)
        best_key = DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE
        best_distance = float("inf")
        for key, candidate in CONFIDENCE_UNCERTAINTY_PROFILE_VALUES.items():
            distance = abs(float(candidate) - numeric)
            if distance < best_distance:
                best_key = str(key)
                best_distance = float(distance)
        return best_key

    def _selected_polygon_compare_profile(self) -> str:
        return str(self.polygon_compare_profile_combo.currentData() or DEFAULT_POLYGON_COMPARE_PROFILE)

    def _polygon_compare_values_for_profile(self, profile_key: str | None) -> tuple[float, int] | None:
        profile = str(profile_key or DEFAULT_POLYGON_COMPARE_PROFILE)
        values = POLYGON_COMPARE_PROFILE_VALUES.get(profile)
        if values is None:
            return None
        mask_threshold, boundary_radius = values
        return float(mask_threshold), int(boundary_radius)

    def _polygon_compare_profile_for_values(self, mask_threshold: float | None, boundary_radius: int | None) -> str:
        if mask_threshold is None or boundary_radius is None or not isfinite(float(mask_threshold)):
            return DEFAULT_POLYGON_COMPARE_PROFILE
        numeric_threshold = float(mask_threshold)
        numeric_radius = int(boundary_radius)
        best_key = DEFAULT_POLYGON_COMPARE_PROFILE
        best_distance = float("inf")
        for key, (candidate_threshold, candidate_radius) in POLYGON_COMPARE_PROFILE_VALUES.items():
            distance = abs(float(candidate_threshold) - numeric_threshold) + abs(int(candidate_radius) - numeric_radius)
            if distance < best_distance:
                best_key = str(key)
                best_distance = float(distance)
        return best_key
