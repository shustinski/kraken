"""Presenter for Karakal."""

from __future__ import annotations

import json
import logging
import re
import shutil
from bisect import bisect_right
from dataclasses import replace
from datetime import datetime
from heapq import heappush, heapreplace
from math import isfinite
from pathlib import Path
from time import perf_counter

from PyQt6.QtCore import QObject, QSignalBlocker, QThread, QTimer, Qt
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from kraken_core.analysis_protocol import (
    AnalysisParameter,
    AnalysisProfileKind,
    AnalysisScaleMode,
    AnalysisSourceRole,
)

from ..core.analysis_modes import (
    CONFIDENCE_COMPARISON_MODE,
    INTRA_MODEL_CONFIDENCE_MODE,
    INTER_MODEL_ANALYSIS_MODE,
    MODEL_OUTPUT_CONFIDENCE_MODE,
    POINT_OBJECT_TYPE,
    POLYGON_OBJECT_TYPE,
    confidence_metric_family,
    default_confidence_model_id,
    default_metric_key,
    display_metric_keys,
    geometry_mode_for_object_type,
    metric_level_key,
    metric_visual_ratio,
    object_type_from_geometry_mode,
    percentile_basis_keys,
    resolve_analysis_context,
)
from ..core.analysis_profiles import (
    DEFAULT_ANALYSIS_PROFILE,
    AnalysisPreflightReport,
    analysis_profile_definition,
    build_standalone_preflight,
)
from ..core.backend_constants import (
    BCE_SCORE_CAP,
)
from ..core.domain import (
    BuildOptions,
    BuildResult,
    ComparisonPairSelection,
    ComparisonTarget,
    FolderSpec,
    FrameIdentity,
    FrameRecord,
    ModelSpec,
)
from ..core.grid_anomaly import (
    GridCellReferenceProfile,
    GridDamageAnalysisConfig,
    build_grid_cell_reference_profile_path,
)
from ..core.image_formats import SUPPORTED_IMAGE_EXTENSION_SET
from ..core.project_profile import AnalysisSourceBinding, KarakalAnalysisProfileV1, SourceBindingKind
from ..core.exports import (
    available_result_layer_exports,
    export_grid_cell_defect_bmps,
    export_grid_cell_defect_canvas,
    export_record_assets,
    export_result_layer_jpgs,
    export_result_layers_jpgs,
)
from ..core.metric_keys import (
    _parse_model_metric_key,
    combined_pair_metric_key,
    confidence_pair_metric_key,
    compute_metric_percentiles,
    metric_higher_is_better,
    metric_value_for_record,
    pair_metric_key,
    parse_combined_pair_metric_key,
    parse_confidence_pair_metric_key,
    parse_pair_metric_key,
)
from ..core.workers import (
    AnalyticsWorker,
    FrameIndexWorker,
    GridInspectionWorker,
    PairedGridInspectionWorker,
    WorkerBase,
)
from ..infra.services import KarakalSettingsService
from ..ui.details_dialog import ExtendFrameDetailsDialog
from ..ui.matrix_view import MatrixLayoutConfig, build_matrix_layout
from ..ui.ui_components import FolderRowWidget
from ..ui.ui_constants import (
    DEFAULT_CELL_SIZE,
    DEFAULT_BOUNDARY_RADIUS,
    DEFAULT_COMPARISON_TARGET,
    DEFAULT_CONFIDENCE_UNCERTAINTY_DELTA,
    DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE,
    DEFAULT_FRAMES_PER_ROW,
    DEFAULT_GEOMETRY_MODE,
    DEFAULT_GRADIENT_NAME,
    DEFAULT_MATRIX_COLUMNS,
    DEFAULT_MATRIX_LAYOUT_MODE,
    DEFAULT_MATRIX_SCORE_VIEW_MODE,
    DEFAULT_MATRIX_METRIC_KEY,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_MATRIX_ROWS,
    GRID_INSPECTION_DEFAULT_ERROR_TYPES,
    GRID_INSPECTION_DAMAGE_METRIC_KEY,
    GRID_INSPECTION_ERROR_TYPE_OPTIONS,
    GRID_INSPECTION_FIXED_TUNING,
    grid_inspection_error_type_icon,
    DEFAULT_POINT_CONFIDENCE_RADIUS,
    DEFAULT_POINT_EXTRACTION_MODE,
    DEFAULT_POLYGON_CONFIDENCE_SUMMARY,
    DEFAULT_POLYGON_COMPARE_PROFILE,
    DEFAULT_TOTAL_FRAMES,
    FOLDER_CHECKED_ROLE,
    FOLDER_CONFIDENCE_ROLE,
    FOLDER_CONFIDENCE_EXPANDED_ROLE,
    FOLDER_LABEL_ROLE,
    FOLDER_ROW_MIN_HEIGHT,
    MATRIX_METRIC_OPTIONS,
    PERCENTILE_BAND_BOUNDS,
    PERCENTILE_BAND_TITLES,
    CONFIDENCE_UNCERTAINTY_PROFILE_VALUES,
    POLYGON_COMPARE_PROFILE_VALUES,
)
from .state import ExtendMatrixTabState

PAIR_ROLE_MODEL_IDS = int(Qt.ItemDataRole.UserRole) + 40
PAIR_OPERATION_ORDER = ("xor", "iou", "dice")
PAIR_OPERATION_LABELS = {"xor": "XOR", "iou": "IoU", "dice": "Dice"}
PAIR_OPERATION_SHORT_LABELS = {"xor": "X", "iou": "I", "dice": "D"}
GRID_INSPECTION_ERROR_LIST_LIMIT = 1000
_LOGGER = logging.getLogger(__name__)


class KarakalPresenter(QObject):
    """Coordinate UI state, background workers and matrix tabs."""

    def __init__(self, view, settings_service: KarakalSettingsService) -> None:
        super().__init__(view)
        self._view = view
        self._settings_service = settings_service
        self._worker_thread: QThread | None = None
        self._worker = None
        self._worker_kind: str | None = None
        self._active_compute_state: ExtendMatrixTabState | None = None
        self._original_folder: FolderSpec | None = None
        self._export_folder: Path | None = None
        self._saved_validation_mask_payload: dict[str, object] = (
            self._settings_service.load_validation_mask_payload() or {}
        )
        self._folder_check_guard = False
        self._pair_matrix_guard = False
        self._pair_defaults_initialized = False
        self._comparison_pair_operations: dict[tuple[str, str], set[str]] = {}
        self._pair_operation_buttons: dict[tuple[str, str, str], QPushButton] = {}
        self._tab_states: dict[object, ExtendMatrixTabState] = {}
        self._pending_build_snapshot: dict[str, object] | None = None
        self._details_dialogs: list[ExtendFrameDetailsDialog] = []
        self._details_view_payload: dict[str, object] = self._settings_service.load_details_view_payload() or {}
        self._request_generation = 0
        self._active_request_generation: int | None = None
        self._active_processing_keys: set[str] = set()
        self._active_grid_inspection_partial_keys: set[str] | None = None
        self._app_mode_refresh_generation = 0
        self._active_progress_current = 0
        self._active_progress_total = 0
        self._active_progress_key = ""
        self._deferred_analytics_restart: tuple[ExtendMatrixTabState, bool] | None = None
        self._histogram_update_generation = 0
        self._last_active_tab_state: ExtendMatrixTabState | None = None
        self._analysis_profile = DEFAULT_ANALYSIS_PROFILE
        self._preflight_signature: tuple[object, ...] | None = None
        self._preflight_report: AnalysisPreflightReport | None = None
        self._auto_compute_after_build = False
        self._auto_compute_state_after_cleanup: ExtendMatrixTabState | None = None

    def __getattr__(self, name: str):
        return getattr(self._view, name)

    def _current_tab_state(self) -> ExtendMatrixTabState | None:
        widget = self.matrix_tabs.currentWidget()
        if widget is None:
            if self._last_active_tab_state is not None and any(
                candidate is self._last_active_tab_state for candidate in self._tab_states.values()
            ):
                return self._last_active_tab_state
            return None
        state = self._tab_states.get(widget)
        if state is not None:
            return state
        if self._last_active_tab_state is not None and any(
            candidate is self._last_active_tab_state for candidate in self._tab_states.values()
        ):
            return self._last_active_tab_state
        return None

    @staticmethod
    def _set_row_visible(row: object | None, visible: bool) -> None:
        if row is not None and hasattr(row, "setVisible"):
            row.setVisible(bool(visible))

    def _selected_analysis_mode(self) -> str:
        return str(self.analysis_mode_combo.currentData() or INTER_MODEL_ANALYSIS_MODE)

    def _selected_grid_error_types(self) -> tuple[str, ...]:
        checks = getattr(self, "grid_error_type_checks", {}) or {}
        selected: list[str] = []
        for _label_key, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS:
            checkbox = checks.get(str(error_type)) if isinstance(checks, dict) else None
            if checkbox is None:
                selected.append(str(error_type))
                continue
            try:
                if checkbox.isChecked():
                    selected.append(str(error_type))
            except Exception:
                selected.append(str(error_type))
        return tuple(selected)

    @staticmethod
    def _normalize_grid_error_types(value: object) -> tuple[str, ...]:
        allowed = tuple(str(error_type) for _label_key, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS)
        allowed_set = set(allowed)
        if value is None:
            return tuple(GRID_INSPECTION_DEFAULT_ERROR_TYPES)
        if isinstance(value, str):
            raw_values = (value,)
        else:
            try:
                raw_values = tuple(value)  # type: ignore[arg-type]
            except Exception:
                return tuple(GRID_INSPECTION_DEFAULT_ERROR_TYPES)
        return tuple(
            error_type
            for error_type in allowed
            if error_type in {str(item) for item in raw_values} and error_type in allowed_set
        )

    def _grid_inspection_config_payload(self) -> dict[str, object]:
        payload: dict[str, object] = dict(GRID_INSPECTION_FIXED_TUNING)
        payload["enabled_error_types"] = list(self._selected_grid_error_types())
        return payload

    def _set_grid_inspection_config_controls(self, payload: dict[str, object] | None) -> None:
        values = dict(payload or {})
        enabled_error_types = self._normalize_grid_error_types(values.get("enabled_error_types"))
        checks = getattr(self, "grid_error_type_checks", {}) or {}
        if isinstance(checks, dict):
            for _label_key, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS:
                checkbox = checks.get(str(error_type))
                if checkbox is None:
                    continue
                blocker = QSignalBlocker(checkbox)
                checkbox.setChecked(str(error_type) in enabled_error_types)
                del blocker

    @staticmethod
    def _grid_damage_config_from_payload(payload: dict[str, object] | None) -> GridDamageAnalysisConfig:
        fixed_tuning = dict(GRID_INSPECTION_FIXED_TUNING)
        strictness = fixed_tuning["strictness"] / 100.0
        fill = fixed_tuning["fill_sensitivity"] / 100.0
        merge = fixed_tuning["merge_sensitivity"] / 100.0
        noise = fixed_tuning["noise_filter"] / 100.0
        threshold = fixed_tuning["defect_threshold"] / 100.0
        enabled_error_types = KarakalPresenter._normalize_grid_error_types(
            (payload or {}).get("enabled_error_types")
        )
        strict_boost = strictness - 0.5
        return GridDamageAnalysisConfig(
            include_debug_payload=False,
            debug=False,
            min_contour_area=4.0 + 20.0 * noise,
            min_cell_size=max(2, int(round(2.0 + 5.0 * noise))),
            filled_ratio_delta=max(0.04, 0.35 - 0.26 * fill - 0.08 * strict_boost),
            bad_score_threshold=max(0.30, min(0.98, threshold - 0.10 * strict_boost)),
            merged_size_ratio=max(1.10, 1.87 - 0.58 * merge - 0.14 * strict_boost),
            merged_area_ratio=max(1.10, 1.86 - 0.62 * merge - 0.14 * strict_boost),
            enabled_reason_types=enabled_error_types,
        ).normalized()

    def _selected_grid_damage_config(self) -> GridDamageAnalysisConfig:
        return self._grid_damage_config_from_payload(self._grid_inspection_config_payload())

    @staticmethod
    def _grid_inspection_layer_keys() -> tuple[str, ...]:
        return ("confidence", "binary", "comparison")

    def _grid_inspection_views(self) -> dict[str, object]:
        views = getattr(self, "grid_inspection_matrix_views", None)
        if isinstance(views, dict) and views:
            return {str(key): value for key, value in views.items()}
        view = getattr(self, "grid_inspection_matrix_view", None)
        return {"confidence": view} if view is not None else {}

    def _grid_inspection_model_id_for_state(self, state: ExtendMatrixTabState) -> str | None:
        current = str(getattr(state, "grid_inspection_model_id", "") or "")
        valid_ids = {str(spec.model_id) for spec in tuple(state.build_result.model_specs or ())}
        if current in valid_ids:
            return current
        current_item = self.folder_list.currentItem() if hasattr(self, "folder_list") else None
        if current_item is not None:
            selected = self._model_id_for_folder_item(current_item, state.build_result)
            if selected in valid_ids:
                return str(selected)
        for spec in tuple(state.build_result.model_specs or ()):
            model_id = str(spec.model_id)
            if any(
                bool((record.model_mask_paths or {}).get(model_id))
                and bool((record.model_prob_paths or {}).get(model_id))
                for record in tuple(state.build_result.records or ())
            ):
                return model_id
        specs = tuple(state.build_result.model_specs or ())
        return str(specs[0].model_id) if specs else None

    def _on_grid_inspection_layer_changed(self, layer_key: str) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        normalized = str(layer_key) if str(layer_key) in self._grid_inspection_layer_keys() else "confidence"
        state.grid_inspection_layer = normalized
        layers = getattr(state, "grid_inspection_payloads_by_layer", {}) or {}
        state.grid_inspection_payload_by_key = dict(layers.get(normalized, {}) or {})
        self._refresh_grid_inspection_errors_panel(state)
        self._schedule_metric_histogram_update(state)

    def _sync_grid_inspection_layer_tabs(self, state: ExtendMatrixTabState | None) -> None:
        tabs = getattr(self, "grid_inspection_layer_tabs", None)
        if tabs is None:
            return
        keys = self._grid_inspection_layer_keys()
        available = {key: False for key in keys}
        if state is not None and bool(getattr(state, "grid_inspection_results_ready", False)):
            layers = getattr(state, "grid_inspection_payloads_by_layer", {}) or {}
            available = {key: bool(layers.get(key)) for key in keys}
        elif state is not None:
            model_id = self._grid_inspection_model_id_for_state(state)
            has_binary = bool(
                model_id
                and any(
                    bool((record.model_mask_paths or {}).get(str(model_id))) for record in state.build_result.records
                )
            )
            has_confidence = bool(
                model_id
                and any(
                    bool((record.model_prob_paths or {}).get(str(model_id))) for record in state.build_result.records
                )
            )
            available["confidence"] = has_confidence or has_binary
            available["binary"] = has_binary
            available["comparison"] = has_binary and has_confidence
        if not any(available.values()):
            available["confidence"] = True
        for index, key in enumerate(keys):
            tabs.setTabEnabled(index, bool(available[key]))
        current_key = (
            str(getattr(state, "grid_inspection_layer", "confidence") or "confidence")
            if state is not None
            else "confidence"
        )
        if not available.get(current_key, False):
            current_key = next(key for key in keys if available[key])
            if state is not None:
                state.grid_inspection_layer = current_key
            tabs.setCurrentIndex(keys.index(current_key))

    def _selected_comparison_target(self) -> ComparisonTarget:
        value = str(self.comparison_target_combo.currentData() or DEFAULT_COMPARISON_TARGET)
        try:
            return ComparisonTarget(value)
        except Exception:
            return ComparisonTarget.OUTPUTS

    @staticmethod
    def _is_dynamic_pair_metric_key(metric_key: str | None) -> bool:
        text = str(metric_key or "")
        return (
            parse_pair_metric_key(text) is not None
            or parse_confidence_pair_metric_key(text) is not None
            or parse_combined_pair_metric_key(text) is not None
        )

    def _required_model_count_for_active_mode(self, analysis_mode: str | None = None) -> int:
        mode = str(analysis_mode or self._selected_analysis_mode() or INTER_MODEL_ANALYSIS_MODE)
        return 2 if mode in {INTER_MODEL_ANALYSIS_MODE, CONFIDENCE_COMPARISON_MODE} else 1

    def _required_model_count_for_build(self) -> int:
        return 1

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
        active_build_result = (
            build_result if build_result is not None else (state.build_result if state is not None else None)
        )
        analysis_mode = state.analysis_mode if state is not None else self._selected_analysis_mode()
        object_type = state.object_type if state is not None else self._selected_object_type()
        confidence_model_id = (
            state.confidence_model_id if state is not None else self._selected_confidence_model_id(active_build_result)
        )
        return resolve_analysis_context(
            active_build_result,
            analysis_mode,
            object_type,
            confidence_model_id=confidence_model_id,
        )

    def _excluded_record_keys_for_state(self, state: ExtendMatrixTabState | None) -> set[str]:
        if state is None:
            return set()
        return {str(key) for key in getattr(state, "excluded_record_keys", set()) if str(key)}

    def _record_is_excluded(self, state: ExtendMatrixTabState | None, record: FrameRecord | None) -> bool:
        if state is None or record is None:
            return False
        return str(record.key) in self._excluded_record_keys_for_state(state)

    def _available_metric_keys_for_state(
        self, state: ExtendMatrixTabState | None, build_result: BuildResult | None = None
    ) -> set[str]:
        source = build_result if build_result is not None else (state.build_result if state is not None else None)
        available = getattr(source, "available_metric_keys", None) if source is not None else None
        if available is None:
            return set()
        return {str(key) for key in available if str(key)}

    def _display_metric_keys_for_state(
        self, state: ExtendMatrixTabState | None, build_result: BuildResult | None = None
    ) -> tuple[str, ...]:
        context = self._analysis_context_for_state(state, build_result)
        available = self._available_metric_keys_for_state(state, build_result)
        if build_result is None:
            return tuple(display_metric_keys(context))
        if not available:
            return ()
        keys = [key for key in display_metric_keys(context) if key in available]
        if context.analysis_mode == INTER_MODEL_ANALYSIS_MODE:
            for key in getattr(build_result, "available_metric_keys", ()) or ():
                key_text = str(key)
                if (
                    (
                        parse_pair_metric_key(key_text) is not None
                        or parse_confidence_pair_metric_key(key_text) is not None
                        or parse_combined_pair_metric_key(key_text) is not None
                    )
                    and key_text in available
                    and key_text not in keys
                ):
                    keys.append(key_text)
        return tuple(keys)

    def _percentile_basis_keys_for_state(
        self, state: ExtendMatrixTabState | None, build_result: BuildResult | None = None
    ) -> tuple[str, ...]:
        if self._current_app_mode() == "grid_inspection":
            if state is not None and bool(getattr(state, "grid_inspection_payload_by_key", {}) or {}):
                return (GRID_INSPECTION_DAMAGE_METRIC_KEY,)
            return ()
        context = self._analysis_context_for_state(state, build_result)
        available = self._available_metric_keys_for_state(state, build_result)
        if build_result is None:
            return tuple(percentile_basis_keys(context))
        if not available:
            return ()
        keys = [key for key in percentile_basis_keys(context) if key in available]
        if context.analysis_mode == INTER_MODEL_ANALYSIS_MODE:
            for key in getattr(build_result, "available_metric_keys", ()) or ():
                key_text = str(key)
                if (
                    (
                        parse_pair_metric_key(key_text) is not None
                        or parse_confidence_pair_metric_key(key_text) is not None
                        or parse_combined_pair_metric_key(key_text) is not None
                    )
                    and key_text in available
                    and key_text not in keys
                ):
                    keys.append(key_text)
        return tuple(keys)

    def _default_metric_key_for_state(
        self, state: ExtendMatrixTabState | None, build_result: BuildResult | None = None
    ) -> str:
        context = self._analysis_context_for_state(state, build_result)
        if context.analysis_mode == INTER_MODEL_ANALYSIS_MODE:
            target_metric = self._default_metric_key_for_comparison_target(build_result)
            if target_metric is not None:
                return target_metric
        key = default_metric_key(context)
        available = self._available_metric_keys_for_state(state, build_result)
        if not available or key in available:
            return key
        for candidate in self._percentile_basis_keys_for_state(state, build_result):
            if candidate in available:
                return candidate
        for candidate in ("overall_frame_score", "export_priority_score", "model_model_score", "disagreement_score"):
            if candidate in available:
                return candidate
        return next(iter(sorted(available)), "overall_frame_score")

    def _default_metric_key_for_comparison_target(self, build_result: BuildResult | None) -> str | None:
        pairs = self._selected_comparison_pairs()
        pair = pairs[0] if pairs else None
        if pair is None and build_result is not None:
            option_pairs = tuple(getattr(build_result.options, "comparison_pairs", ()) or ())
            pair = option_pairs[0] if option_pairs else None
        if pair is None:
            return None
        available = set(getattr(build_result, "available_metric_keys", ()) or ())
        target = self._selected_comparison_target()
        if target == ComparisonTarget.CONFIDENCE:
            if build_result is not None and not self._pair_has_output_confidence(
                pair.model_a_id, pair.model_b_id, build_result
            ):
                return None
            key = confidence_pair_metric_key(pair.model_a_id, pair.model_b_id, "mae")
        elif target == ComparisonTarget.BOTH:
            if build_result is not None and not self._pair_has_output_confidence(
                pair.model_a_id, pair.model_b_id, build_result
            ):
                return None
            key = combined_pair_metric_key(pair.model_a_id, pair.model_b_id)
        else:
            operation = next(
                (candidate for candidate in ("dice", "iou", "xor") if candidate in pair.operations),
                str(pair.operations[0]),
            )
            key = pair_metric_key(pair.model_a_id, pair.model_b_id, operation)
        if not available or key in available or bool(getattr(build_result, "scores_computed", False)):
            return key
        return None

    @staticmethod
    def _is_base_only_build_result(build_result: BuildResult | None) -> bool:
        if build_result is None:
            return False
        specs = tuple(getattr(build_result, "model_specs", ()) or ())
        return len(specs) == 1 and str(getattr(specs[0], "model_id", "") or "") == "base_layer"

    def _fallback_metric_keys_for_build_result(self, build_result: BuildResult | None) -> list[str]:
        available = set(build_result.available_metric_keys if build_result is not None else ())
        candidates = ("overall_frame_score", "export_priority_score", "model_model_score", "disagreement_score")
        keys = [key for key in candidates if not available or key in available]
        return keys or [next(iter(sorted(available)), "overall_frame_score")]

    def _confidence_context_available(self, context, build_result: BuildResult | None) -> bool:
        return (
            context.analysis_mode in {INTRA_MODEL_CONFIDENCE_MODE, MODEL_OUTPUT_CONFIDENCE_MODE}
            and context.confidence_model_id is not None
        )

    def _has_model_output_confidence_maps(self, build_result: BuildResult | None = None) -> bool:
        if build_result is not None:
            for spec in getattr(build_result, "model_specs", ()) or ():
                if spec.prob_folder is None:
                    continue
                model_id = str(spec.model_id)
                if any(bool((record.model_prob_paths or {}).get(model_id)) for record in build_result.records):
                    return True
            if getattr(build_result, "model_specs", None):
                return False
        for spec in self._checked_model_specs():
            if spec.prob_folder is not None:
                return True
        return False

    def _has_model_output_confidence_pair(self, build_result: BuildResult | None = None) -> bool:
        if build_result is not None:
            model_ids: set[str] = set()
            for spec in getattr(build_result, "model_specs", ()) or ():
                if spec.prob_folder is None:
                    continue
                model_id = str(spec.model_id)
                if any(bool((record.model_prob_paths or {}).get(model_id)) for record in build_result.records):
                    model_ids.add(model_id)
            return len(model_ids) >= 2
        return sum(1 for spec in self._checked_model_specs() if spec.prob_folder is not None) >= 2

    def _model_has_output_confidence(self, model_id: str, build_result: BuildResult | None = None) -> bool:
        model_id = str(model_id)
        if build_result is not None:
            for spec in getattr(build_result, "model_specs", ()) or ():
                if str(spec.model_id) == model_id and spec.prob_folder is not None:
                    return True
            return any(bool((record.model_prob_paths or {}).get(model_id)) for record in build_result.records)
        for spec in self._checked_model_specs():
            if str(spec.model_id) == model_id:
                return spec.prob_folder is not None
        return False

    def _pair_has_output_confidence(self, model_a: str, model_b: str, build_result: BuildResult | None = None) -> bool:
        return self._model_has_output_confidence(model_a, build_result) and self._model_has_output_confidence(
            model_b, build_result
        )

    def _selected_target_requires_pair_confidence(self) -> bool:
        return self._selected_comparison_target() in {ComparisonTarget.CONFIDENCE, ComparisonTarget.BOTH}

    def _set_analysis_mode_option_enabled(self, mode_key: str, enabled: bool, tooltip: str = "") -> None:
        index = self.analysis_mode_combo.findData(str(mode_key))
        if index < 0:
            return
        item = self.analysis_mode_combo.model().item(index)
        if item is not None:
            if str(mode_key) == MODEL_OUTPUT_CONFIDENCE_MODE:
                base_text = self._t("analysis.mode.model_output_confidence")
                item.setText(base_text)
            elif str(mode_key) == CONFIDENCE_COMPARISON_MODE:
                base_text = self._t("analysis.mode.confidence_comparison")
                item.setText(base_text)
            item.setEnabled(bool(enabled))
            item.setToolTip(str(tooltip or ""))
            item.setForeground(QBrush(QColor("#edf3fb" if enabled else "#6f7a86")))
            font = item.font()
            font.setItalic(not bool(enabled))
            item.setFont(font)

    def _set_comparison_target_option_enabled(self, target: ComparisonTarget, enabled: bool, tooltip: str = "") -> None:
        if not hasattr(self, "comparison_target_combo"):
            return
        index = self.comparison_target_combo.findData(target.value)
        if index < 0:
            return
        item = self.comparison_target_combo.model().item(index)
        if item is None:
            return
        label_key = {
            ComparisonTarget.OUTPUTS: "comparison_target.outputs",
            ComparisonTarget.CONFIDENCE: "comparison_target.confidence",
            ComparisonTarget.BOTH: "comparison_target.both",
        }[target]
        item.setText(self._t(label_key))
        item.setEnabled(bool(enabled))
        item.setToolTip(str(tooltip or ""))
        item.setForeground(QBrush(QColor("#edf3fb" if enabled else "#6f7a86")))
        font = item.font()
        font.setItalic(not bool(enabled))
        item.setFont(font)

    def _sync_confidence_map_function_state(
        self, build_result: BuildResult | None = None, *, allow_fallback: bool = True
    ) -> bool:
        available = self._has_model_output_confidence_maps(build_result)
        pair_available = self._has_model_output_confidence_pair(build_result)
        tooltip = "" if available else self._t("hint.model_output_confidence_unavailable")
        self._set_analysis_mode_option_enabled(MODEL_OUTPUT_CONFIDENCE_MODE, available, tooltip)
        pair_tooltip = "" if pair_available else self._t("hint.confidence_comparison_unavailable")
        self._set_analysis_mode_option_enabled(CONFIDENCE_COMPARISON_MODE, pair_available, pair_tooltip)
        self._set_comparison_target_option_enabled(ComparisonTarget.OUTPUTS, True, "")
        self._set_comparison_target_option_enabled(ComparisonTarget.CONFIDENCE, pair_available, pair_tooltip)
        self._set_comparison_target_option_enabled(ComparisonTarget.BOTH, pair_available, pair_tooltip)
        if allow_fallback and (
            (not available and self._selected_analysis_mode() == MODEL_OUTPUT_CONFIDENCE_MODE)
            or (not pair_available and self._selected_analysis_mode() == CONFIDENCE_COMPARISON_MODE)
            or (
                not pair_available
                and self._selected_comparison_target() in {ComparisonTarget.CONFIDENCE, ComparisonTarget.BOTH}
            )
        ):
            if self._selected_analysis_mode() in {MODEL_OUTPUT_CONFIDENCE_MODE, CONFIDENCE_COMPARISON_MODE}:
                blocker = QSignalBlocker(self.analysis_mode_combo)
                fallback_index = self.analysis_mode_combo.findData(INTER_MODEL_ANALYSIS_MODE)
                self.analysis_mode_combo.setCurrentIndex(fallback_index if fallback_index >= 0 else 0)
                del blocker
            if self._selected_comparison_target() in {ComparisonTarget.CONFIDENCE, ComparisonTarget.BOTH}:
                blocker = QSignalBlocker(self.comparison_target_combo)
                fallback_index = self.comparison_target_combo.findData(ComparisonTarget.OUTPUTS.value)
                self.comparison_target_combo.setCurrentIndex(fallback_index if fallback_index >= 0 else 0)
                del blocker
                self._apply_comparison_target_to_states()
            self._apply_global_analysis_context_to_all_states()
        return available

    def _sync_mode_controls(
        self, state: ExtendMatrixTabState | None = None, build_result: BuildResult | None = None
    ) -> None:
        self._sync_confidence_map_function_state(build_result)
        context = self._analysis_context_for_state(state, build_result)
        is_grid_inspection_mode = self._current_app_mode() == "grid_inspection"
        is_confidence_mode = context.analysis_mode in {INTRA_MODEL_CONFIDENCE_MODE, MODEL_OUTPUT_CONFIDENCE_MODE}
        is_confidence_comparison = context.analysis_mode == CONFIDENCE_COMPARISON_MODE
        has_scope_choices = bool(getattr(build_result, "model_specs", ()) if build_result is not None else ())
        is_confidence = self._confidence_context_available(context, build_result)
        is_point = context.object_type == POINT_OBJECT_TYPE
        if hasattr(self, "metric_settings_group"):
            self.metric_settings_group.setVisible(not is_grid_inspection_mode)
        self._set_row_visible(getattr(self, "_matrix_pixel_size_row", None), False)
        self._set_row_visible(getattr(self, "_matrix_layout_row", None), False)
        self._set_row_visible(getattr(self, "_matrix_total_frames_row", None), False)
        self._set_row_visible(getattr(self, "_matrix_rows_row", None), False)
        self._set_row_visible(getattr(self, "_matrix_columns_row", None), False)
        self._set_row_visible(
            getattr(self, "_matrix_comparison_target_row", None), context.analysis_mode == INTER_MODEL_ANALYSIS_MODE
        )
        self._set_row_enabled(getattr(self, "_matrix_frames_per_row_row", None), True)
        self._set_row_visible(getattr(self, "_matrix_frames_per_row_row", None), True)
        self._set_row_visible(
            getattr(self, "_metric_scope_row", None), (not is_grid_inspection_mode) and (not is_confidence_comparison)
        )
        self._set_row_enabled(
            getattr(self, "_metric_scope_row", None),
            has_scope_choices and (not is_grid_inspection_mode) and (not is_confidence_comparison),
        )
        self._set_row_visible(
            getattr(self, "_metric_select_row", None), (not is_grid_inspection_mode) and (not is_confidence_mode)
        )
        self._set_row_visible(getattr(self, "_matrix_confidence_delta_row", None), is_confidence)
        self._set_row_visible(
            getattr(self, "_matrix_polygon_confidence_summary_row", None), is_confidence and not is_point
        )
        self._set_row_visible(
            getattr(self, "_matrix_polygon_compare_profile_row", None),
            not is_confidence and not is_confidence_comparison and not is_point,
        )
        self._set_row_visible(
            getattr(self, "_matrix_point_radius_row", None),
            not is_confidence and not is_confidence_comparison and is_point,
        )
        self._set_row_visible(getattr(self, "_matrix_point_confidence_radius_row", None), is_confidence and is_point)
        self._set_row_visible(getattr(self, "_matrix_point_mode_row", None), is_point)
        self._set_row_visible(getattr(self, "_matrix_frame_type_filter_row", None), False)
        if hasattr(self, "_grid_inspection_tuning_group"):
            self._grid_inspection_tuning_group.setVisible(self._current_app_mode() == "grid_inspection")
        self._sync_grid_reference_controls(state if is_grid_inspection_mode else None)
        if hasattr(self, "pair_matrix_group"):
            self.pair_matrix_group.setVisible(
                context.analysis_mode == INTER_MODEL_ANALYSIS_MODE
                and self._analysis_profile == AnalysisProfileKind.MODEL_COMPARISON
            )
            self._refresh_pair_matrix()

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
            specs.append(
                ModelSpec(
                    model_id=model_id,
                    display_name=label,
                    mask_folder=folder_path,
                    prob_folder=confidence_folder,
                    threshold=threshold,
                )
            )
        return tuple(specs)

    def _analysis_preflight_signature_value(self) -> tuple[object, ...]:
        specs = self._checked_model_specs()
        return (
            self._analysis_profile.value,
            str(self._original_folder.path) if self._original_folder is not None else "",
            tuple(
                (
                    spec.model_id,
                    str(spec.mask_folder),
                    str(spec.prob_folder) if spec.prob_folder is not None else "",
                )
                for spec in specs
            ),
        )

    def _refresh_analysis_preflight(self, *, force: bool = False) -> AnalysisPreflightReport:
        signature = self._analysis_preflight_signature_value()
        if force or signature != self._preflight_signature or self._preflight_report is None:
            self._preflight_report = build_standalone_preflight(
                self._analysis_profile,
                self._original_folder,
                self._checked_model_specs(),
            )
            self._preflight_signature = signature
        report = self._preflight_report
        if hasattr(self._view, "set_analysis_preflight"):
            self._view.set_analysis_preflight(report)
        return report

    def _analysis_profile_availability(self) -> dict[AnalysisProfileKind, tuple[bool, str]]:
        specs = self._checked_model_specs()
        model_count = len(specs)
        confidence_count = sum(1 for spec in specs if spec.prob_folder is not None)
        has_grid_source = self._original_folder is not None or model_count > 0
        return {
            AnalysisProfileKind.MODEL_COMPARISON: (
                model_count >= 2,
                "" if model_count >= 2 else self._t("profile.unavailable.models"),
            ),
            AnalysisProfileKind.CONFIDENCE_AUDIT: (
                model_count >= 1 and confidence_count >= 1,
                "" if model_count >= 1 and confidence_count >= 1 else self._t("profile.unavailable.confidence"),
            ),
            AnalysisProfileKind.GRID_DEFECTS: (
                has_grid_source,
                "" if has_grid_source else self._t("profile.unavailable.grid"),
            ),
        }

    def _on_analysis_profile_changed(self, value: str) -> None:
        profile = analysis_profile_definition(value)
        self._analysis_profile = profile.key
        controls = (
            (self.app_mode_combo, profile.app_mode),
            (self.analysis_mode_combo, profile.analysis_mode),
            (self.comparison_target_combo, profile.comparison_target),
        )
        for combo, selected_value in controls:
            blocker = QSignalBlocker(combo)
            index = combo.findData(selected_value)
            combo.setCurrentIndex(index if index >= 0 else 0)
            del blocker
        self._preflight_signature = None
        self._on_app_mode_changed()
        self._sync_mode_controls(self._current_tab_state())
        self._sync_action_buttons()

    def _on_primary_run_requested(self) -> None:
        report = self._refresh_analysis_preflight(force=True)
        if not report.can_run:
            QMessageBox.warning(self._view, self._t("dialog.warning_title"), self._t("preflight.blocked"))
            return
        self._auto_compute_after_build = True
        self._start_build()

    def _active_pair_model_specs(self) -> tuple[ModelSpec, ...]:
        return tuple(spec for spec in self._checked_model_specs() if str(spec.model_id))

    def _ensure_default_pair_selection(self, specs: tuple[ModelSpec, ...]) -> None:
        id_order = {str(spec.model_id): index for index, spec in enumerate(specs)}
        normalized_operations: dict[tuple[str, str], set[str]] = {}
        for (model_a, model_b), operations in self._comparison_pair_operations.items():
            model_a = str(model_a)
            model_b = str(model_b)
            if model_a not in id_order or model_b not in id_order or model_a == model_b:
                continue
            if id_order[model_a] > id_order[model_b]:
                model_a, model_b = model_b, model_a
            valid_operations = {operation for operation in operations if operation in PAIR_OPERATION_ORDER}
            if valid_operations:
                normalized_operations.setdefault((model_a, model_b), set()).update(valid_operations)
        self._comparison_pair_operations = normalized_operations
        if self._comparison_pair_operations or len(specs) < 2 or self._pair_defaults_initialized:
            return
        self._comparison_pair_operations[(str(specs[0].model_id), str(specs[1].model_id))] = set(PAIR_OPERATION_ORDER)
        self._pair_defaults_initialized = True

    def _on_pair_operation_toggled(self, model_a: str, model_b: str, operation: str, checked: bool) -> None:
        if self._pair_matrix_guard:
            return
        key = (str(model_a), str(model_b))
        operations = self._comparison_pair_operations.setdefault(key, set())
        if checked:
            operations.add(str(operation))
        else:
            operations.discard(str(operation))
            if not operations:
                self._comparison_pair_operations.pop(key, None)
        self._refresh_active_pair_list()
        self._apply_pair_options_to_states()
        self._persist_state()
        self._sync_action_buttons()

    def _refresh_pair_matrix(self) -> None:
        if not hasattr(self, "pair_matrix_table"):
            return
        specs = self._active_pair_model_specs()
        self._ensure_default_pair_selection(specs)
        if hasattr(self, "pair_matrix_group"):
            self.pair_matrix_group.setTitle(self._pair_matrix_title())
        table = self.pair_matrix_table
        self._pair_matrix_guard = True
        try:
            self._pair_operation_buttons.clear()
            table.clear()
            row_specs = specs[:-1]
            column_specs = specs[1:]
            table.setRowCount(len(row_specs))
            table.setColumnCount(len(column_specs))
            table.setHorizontalHeaderLabels(
                [self._pair_header_label(index + 1, spec) for index, spec in enumerate(column_specs)]
            )
            table.setVerticalHeaderLabels(
                [self._pair_header_label(index, spec) for index, spec in enumerate(row_specs)]
            )
            for index, spec in enumerate(column_specs):
                header_tooltip = str(spec.display_name or spec.model_id)
                horizontal = table.horizontalHeaderItem(index)
                if horizontal is not None:
                    horizontal.setToolTip(header_tooltip)
            for index, spec in enumerate(row_specs):
                header_tooltip = str(spec.display_name or spec.model_id)
                vertical = table.verticalHeaderItem(index)
                if vertical is not None:
                    vertical.setToolTip(header_tooltip)
            for row, spec_a in enumerate(row_specs):
                for column, spec_b in enumerate(column_specs):
                    source_index = row
                    target_index = column + 1
                    if target_index < source_index:
                        item = QTableWidgetItem("")
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                        table.setItem(row, column, item)
                        continue
                    if target_index == source_index:
                        item = QTableWidgetItem("-")
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                        table.setItem(row, column, item)
                        continue
                    model_a = str(spec_a.model_id)
                    model_b = str(spec_b.model_id)
                    if self._selected_target_requires_pair_confidence() and not self._pair_has_output_confidence(
                        model_a, model_b
                    ):
                        item = QTableWidgetItem("N/A")
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        item.setToolTip(self._t("hint.confidence_comparison_unavailable"))
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                        item.setForeground(QBrush(QColor("#6f7a86")))
                        table.setItem(row, column, item)
                        continue
                    operations = self._comparison_pair_operations.get((model_a, model_b), set())
                    cell = QWidget(table)
                    cell_layout = QHBoxLayout(cell)
                    cell_layout.setContentsMargins(1, 1, 1, 1)
                    cell_layout.setSpacing(1)
                    for operation in PAIR_OPERATION_ORDER:
                        button = QPushButton(PAIR_OPERATION_SHORT_LABELS[operation], cell)
                        button.setCheckable(True)
                        button.setChecked(operation in operations)
                        button.setFixedSize(26, 22)
                        button.setToolTip(
                            f"{spec_a.display_name} -> {spec_b.display_name}: {PAIR_OPERATION_LABELS[operation]}"
                        )
                        button.toggled.connect(
                            lambda checked, a=model_a, b=model_b, op=operation: self._on_pair_operation_toggled(
                                a, b, op, bool(checked)
                            )
                        )
                        self._pair_operation_buttons[(model_a, model_b, operation)] = button
                        cell_layout.addWidget(button)
                    table.setCellWidget(row, column, cell)
            table.resizeRowsToContents()
            row_height = 28
            header_height = max(28, table.horizontalHeader().height())
            table.setMinimumHeight(min(520, max(100, header_height + row_height * max(1, len(row_specs)) + 34)))
        finally:
            self._pair_matrix_guard = False
        self._refresh_active_pair_list()

    @staticmethod
    def _pair_header_label(index: int, spec: ModelSpec) -> str:
        text = str(spec.display_name or spec.model_id)
        compact = text
        if len(compact) > 14:
            compact = f"{compact[:6]}...{compact[-5:]}"
        return f"{index + 1}. {compact}"

    def _pair_target_metric_label(self) -> str:
        target = self._selected_comparison_target()
        if target == ComparisonTarget.CONFIDENCE:
            return "MAE"
        if target == ComparisonTarget.BOTH:
            return "Combined Risk"
        return "Dice"

    def _pair_matrix_title(self) -> str:
        target = self._selected_comparison_target()
        target_label = {
            ComparisonTarget.OUTPUTS: self._t("comparison_target.outputs"),
            ComparisonTarget.CONFIDENCE: self._t("comparison_target.confidence"),
            ComparisonTarget.BOTH: self._t("comparison_target.both"),
        }.get(target, self._t("comparison_target.outputs"))
        pair_count = len(self._selected_comparison_pairs())
        return (
            f"{self._t('pairs.group')} - {target_label} / {self._pair_target_metric_label()}"
            f" · {self._t('pairs.summary', count=pair_count)}"
        )

    def _selected_comparison_pairs(self) -> tuple[ComparisonPairSelection, ...]:
        specs = self._active_pair_model_specs()
        self._ensure_default_pair_selection(specs)
        specs_by_id = {str(spec.model_id): spec for spec in specs}
        pairs: list[ComparisonPairSelection] = []
        for (model_a, model_b), operations in self._comparison_pair_operations.items():
            if model_a not in specs_by_id or model_b not in specs_by_id or model_a == model_b:
                continue
            ordered_operations = tuple(operation for operation in PAIR_OPERATION_ORDER if operation in operations)
            if ordered_operations:
                pairs.append(ComparisonPairSelection(model_a, model_b, ordered_operations))
        return tuple(pairs)

    def _pair_display_name(self, model_id: str) -> str:
        for spec in self._active_pair_model_specs():
            if str(spec.model_id) == str(model_id):
                return str(spec.display_name or spec.model_id)
        state = self._current_tab_state()
        if state is not None:
            for spec in state.build_result.model_specs:
                if str(spec.model_id) == str(model_id):
                    return str(spec.display_name or spec.model_id)
        return str(model_id)

    def _pair_default_metric_key(self, pair: ComparisonPairSelection) -> str:
        target = self._selected_comparison_target()
        if target == ComparisonTarget.CONFIDENCE:
            return confidence_pair_metric_key(pair.model_a_id, pair.model_b_id, "mae")
        if target == ComparisonTarget.BOTH:
            return combined_pair_metric_key(pair.model_a_id, pair.model_b_id)
        operation = next(
            (candidate for candidate in ("dice", "iou", "xor") if candidate in pair.operations), str(pair.operations[0])
        )
        return pair_metric_key(pair.model_a_id, pair.model_b_id, operation)

    def _refresh_active_pair_list(self) -> None:
        if not hasattr(self, "active_pair_list"):
            return
        if hasattr(self, "pair_matrix_group"):
            self.pair_matrix_group.setTitle(self._pair_matrix_title())
        current_key = (
            str(self.active_pair_list.currentItem().data(Qt.ItemDataRole.UserRole))
            if self.active_pair_list.currentItem() is not None
            else ""
        )
        current_pair = (
            self.active_pair_list.currentItem().data(PAIR_ROLE_MODEL_IDS)
            if self.active_pair_list.currentItem() is not None
            else None
        )
        self.active_pair_list.blockSignals(True)
        try:
            self.active_pair_list.clear()
            pairs = self._selected_comparison_pairs()
            if not pairs:
                item = QListWidgetItem(self._t("pairs.none"))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                self.active_pair_list.addItem(item)
                return
            for pair in pairs:
                operation_text = ", ".join(PAIR_OPERATION_LABELS[operation] for operation in pair.operations)
                target_metric = self._pair_target_metric_label()
                text = f"{self._pair_display_name(pair.model_a_id)} -> {self._pair_display_name(pair.model_b_id)}    {target_metric} ({operation_text})"
                item = QListWidgetItem(text)
                metric_key = self._pair_default_metric_key(pair)
                item.setData(Qt.ItemDataRole.UserRole, metric_key)
                item.setData(PAIR_ROLE_MODEL_IDS, (pair.model_a_id, pair.model_b_id))
                item.setToolTip(metric_key)
                if self._selected_target_requires_pair_confidence() and not self._pair_has_output_confidence(
                    pair.model_a_id, pair.model_b_id
                ):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                    item.setToolTip(self._t("hint.confidence_comparison_unavailable"))
                    item.setForeground(QBrush(QColor("#6f7a86")))
                self.active_pair_list.addItem(item)
                if metric_key == current_key or current_pair == (pair.model_a_id, pair.model_b_id):
                    self.active_pair_list.setCurrentItem(item)
        finally:
            self.active_pair_list.blockSignals(False)

    def _apply_comparison_target_to_states(self) -> None:
        target = self._selected_comparison_target()
        for state in tuple(self._tab_states.values()):
            updated_options = replace(state.build_result.options, comparison_target=target)
            if updated_options != state.build_result.options:
                state.build_result = replace(state.build_result, options=updated_options)
                self._invalidate_state_runtime_caches(state, clear_metric_results=True)
                state.last_analytics_request_signature = None

    def _apply_pair_options_to_states(self) -> None:
        comparison_pairs = self._selected_comparison_pairs()
        target = self._selected_comparison_target()
        for state in tuple(self._tab_states.values()):
            updated_options = replace(
                state.build_result.options, comparison_pairs=comparison_pairs, comparison_target=target
            )
            if updated_options != state.build_result.options:
                state.build_result = replace(state.build_result, options=updated_options)
                self._invalidate_state_runtime_caches(state, clear_metric_results=True)
                state.last_analytics_request_signature = None

    def _on_active_pair_item_clicked(self, item: QListWidgetItem) -> None:
        if not bool(item.flags() & Qt.ItemFlag.ItemIsEnabled):
            return
        metric_key = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not metric_key:
            return
        state = self._current_tab_state()
        if state is None:
            return
        if metric_key not in set(state.build_result.available_metric_keys or ()):
            self._apply_pair_options_to_states()
            state.metric_key = metric_key
            if self._worker is None and bool(getattr(state.build_result, "scores_computed", False)):
                self._start_compute_analytics(state=state, sync_context=False)
            return
        self._sync_metric_controls(state.build_result, preferred_metric_key=metric_key, context_state=state)
        state.metric_key = metric_key
        self._apply_metric_to_state(state, metric_key)
        self._sync_action_buttons()

    def _delete_comparison_pair(self, model_a: str, model_b: str) -> None:
        key = (str(model_a), str(model_b))
        if key not in self._comparison_pair_operations:
            return
        self._comparison_pair_operations.pop(key, None)
        self._refresh_pair_matrix()
        self._apply_pair_options_to_states()
        self._persist_state()
        self._sync_action_buttons()

    def _on_active_pair_context_menu(self, pos) -> None:
        if not hasattr(self, "active_pair_list"):
            return
        item = self.active_pair_list.itemAt(pos)
        if item is None:
            return
        self.active_pair_list.setCurrentItem(item)
        pair_data = item.data(PAIR_ROLE_MODEL_IDS)
        if not isinstance(pair_data, tuple) or len(pair_data) != 2:
            return
        menu = QMenu(self._view)
        delete_action = menu.addAction(self._t("pairs.delete"))
        selected_action = menu.exec(self.active_pair_list.mapToGlobal(pos))
        if selected_action is delete_action:
            self._delete_comparison_pair(str(pair_data[0]), str(pair_data[1]))

    def _model_spec_for_folder_item(
        self, item: QListWidgetItem, build_result: BuildResult | None = None
    ) -> ModelSpec | None:
        path_text = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        label_text = str(item.data(FOLDER_LABEL_ROLE) or "").strip()
        item_path = Path(path_text) if path_text else None
        if build_result is not None:
            if item_path is not None:
                for spec in build_result.model_specs:
                    if str(getattr(spec, "mask_folder", "") or "") == path_text:
                        return spec
            for spec in build_result.model_specs:
                if label_text and str(getattr(spec, "display_name", "") or "") == label_text:
                    return spec
            if item_path is not None:
                item_name = item_path.name
                for spec in build_result.model_specs:
                    spec_name = Path(str(getattr(spec, "mask_folder", "") or "")).name
                    if item_name and spec_name == item_name:
                        return spec
        return None

    def _model_id_for_folder_item(self, item: QListWidgetItem, build_result: BuildResult | None = None) -> str | None:
        spec = self._model_spec_for_folder_item(item, build_result)
        if spec is not None:
            return str(spec.model_id)
        path_text = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        label_value = item.data(FOLDER_LABEL_ROLE)
        label_text = str(label_value or (Path(path_text).name if path_text else "")).strip()
        if not label_text and path_text:
            label_text = Path(path_text).name
        model_id = re.sub(r"[^a-zA-Z0-9_]+", "_", label_text.strip().lower()).strip("_")
        return model_id or None

    def _folder_item_has_output_confidence(
        self, item: QListWidgetItem, build_result: BuildResult | None = None, model_id: str | None = None
    ) -> bool:
        confidence_path_text = str(item.data(FOLDER_CONFIDENCE_ROLE) or "").strip()
        if confidence_path_text:
            return True
        if build_result is None or not model_id:
            return False
        spec = self._model_spec_for_folder_item(item, build_result)
        if spec is None:
            spec = next(
                (candidate for candidate in build_result.model_specs if str(candidate.model_id) == str(model_id)), None
            )
        if spec is None or spec.prob_folder is None:
            return False
        return any(bool((record.model_prob_paths or {}).get(str(model_id))) for record in build_result.records)

    def _folder_item_has_model_confidence(
        self, item: QListWidgetItem, build_result: BuildResult | None = None, model_id: str | None = None
    ) -> bool:
        if build_result is None or not model_id:
            return False
        spec = self._model_spec_for_folder_item(item, build_result)
        if spec is None:
            spec = next(
                (candidate for candidate in build_result.model_specs if str(candidate.model_id) == str(model_id)), None
            )
        if spec is None:
            return False
        return True

    def _preferred_model_metric_key_for_folder_item(
        self, state: ExtendMatrixTabState, item: QListWidgetItem, model_id: str
    ) -> str:
        current_metric = str(state.metric_key or state.build_result.selected_metric_key or DEFAULT_MATRIX_METRIC_KEY)
        parsed = confidence_metric_family(current_metric)
        has_output_confidence = self._folder_item_has_output_confidence(item, state.build_result, model_id)
        has_model_confidence = self._folder_item_has_model_confidence(item, state.build_result, model_id)
        if parsed is not None:
            family, _current_model_id = parsed
            if family == "model_output_confidence":
                return (
                    f"model_output_confidence::{model_id}" if has_output_confidence else f"model_confidence::{model_id}"
                )
            return f"{family}::{model_id}"
        if has_output_confidence:
            return f"model_output_confidence::{model_id}"
        if has_model_confidence:
            return f"model_confidence::{model_id}"
        return current_metric

    def _analysis_mode_for_folder_item(self, state: ExtendMatrixTabState, item: QListWidgetItem, model_id: str) -> str:
        current_mode = str(state.analysis_mode or INTER_MODEL_ANALYSIS_MODE)
        has_output_confidence = self._folder_item_has_output_confidence(item, state.build_result, model_id)
        has_model_confidence = self._folder_item_has_model_confidence(item, state.build_result, model_id)
        if current_mode == MODEL_OUTPUT_CONFIDENCE_MODE:
            return MODEL_OUTPUT_CONFIDENCE_MODE if has_output_confidence else INTRA_MODEL_CONFIDENCE_MODE
        if current_mode == INTRA_MODEL_CONFIDENCE_MODE:
            return INTRA_MODEL_CONFIDENCE_MODE
        if has_output_confidence:
            return MODEL_OUTPUT_CONFIDENCE_MODE
        if has_model_confidence:
            return INTRA_MODEL_CONFIDENCE_MODE
        return current_mode

    def _preferred_metric_key_for_folder_item_in_current_mode(
        self, state: ExtendMatrixTabState, item: QListWidgetItem, model_id: str
    ) -> str:
        current_mode = self._selected_analysis_mode()
        current_metric = str(state.metric_key or state.build_result.selected_metric_key or DEFAULT_MATRIX_METRIC_KEY)
        if current_mode == MODEL_OUTPUT_CONFIDENCE_MODE:
            if self._folder_item_has_output_confidence(item, state.build_result, model_id):
                return f"model_output_confidence::{model_id}"
            return current_metric
        if current_mode == INTRA_MODEL_CONFIDENCE_MODE:
            return f"model_confidence::{model_id}"
        return current_metric

    def _on_folder_item_clicked(self, item: QListWidgetItem) -> None:
        state = self._current_tab_state()
        if state is None or self._worker_thread is not None:
            return
        build_result = state.build_result
        model_id = self._model_id_for_folder_item(item, build_result)
        if not model_id:
            return
        if item is not self.folder_list.currentItem():
            self.folder_list.setCurrentItem(item)
        if self._current_app_mode() == "grid_inspection":
            state.grid_inspection_model_id = str(model_id)
            self._set_details_preferred_model_id(str(model_id))
            self._start_compute_grid_inspection(state)
            return
        self._apply_global_analysis_context_to_state(state)
        preferred_metric_key = self._preferred_metric_key_for_folder_item_in_current_mode(state, item, model_id)

        if self._selected_analysis_mode() in {INTRA_MODEL_CONFIDENCE_MODE, MODEL_OUTPUT_CONFIDENCE_MODE}:
            state.confidence_model_id = model_id
            state.metric_scope = model_id

        scope_blocker = QSignalBlocker(self.metric_scope_combo)
        scope_index = self.metric_scope_combo.findData(model_id)
        if scope_index >= 0:
            self.metric_scope_combo.setCurrentIndex(scope_index)
        del scope_blocker

        self._sync_metric_controls(
            build_result,
            preferred_metric_key=preferred_metric_key,
            preferred_scope_key=model_id,
            context_state=state,
        )
        state.confidence_model_id = self._selected_confidence_model_id(build_result)
        state.metric_scope = str(state.confidence_model_id or "")
        metric_key = str(
            self.metric_combo.currentData()
            or preferred_metric_key
            or self._default_metric_key_for_state(state, build_result)
        )
        state.metric_key = metric_key
        self.metric_combo.setToolTip(self._metric_hint_fallback(metric_key, build_result))
        self._apply_metric_to_state(state, metric_key)
        self._sync_action_buttons()
        self._set_details_preferred_model_id(state.metric_scope or state.confidence_model_id or model_id)

    def _effective_model_specs_for_build(self) -> tuple[ModelSpec, ...]:
        specs = self._checked_model_specs()
        if specs:
            return specs
        if self._original_folder is None:
            return specs
        base_path = Path(self._original_folder.path)
        if not base_path.exists():
            return specs
        threshold, _boundary_radius = self._selected_polygon_compare_values()
        fallback_spec = ModelSpec(
            model_id="base_layer",
            display_name=str(self._original_folder.label or base_path.name or "base_layer"),
            mask_folder=base_path,
            prob_folder=None,
            threshold=float(threshold),
        )
        return (fallback_spec,)

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

    def _selected_polygon_compare_values(self) -> tuple[float, int]:
        values = self._polygon_compare_values_for_profile(self._selected_polygon_compare_profile())
        if values is None:
            values = self._polygon_compare_values_for_profile(DEFAULT_POLYGON_COMPARE_PROFILE)
        if values is None:
            return float(DEFAULT_MASK_THRESHOLD), int(DEFAULT_BOUNDARY_RADIUS)
        return values

    @staticmethod
    def _set_row_enabled(row: object | None, enabled: bool) -> None:
        if row is not None and hasattr(row, "setEnabled"):
            row.setEnabled(bool(enabled))

    def _apply_polygon_compare_profile(self, profile_key: str | None) -> None:
        profile = str(profile_key or DEFAULT_POLYGON_COMPARE_PROFILE)
        values = self._polygon_compare_values_for_profile(profile)
        if values is None:
            values = self._polygon_compare_values_for_profile(DEFAULT_POLYGON_COMPARE_PROFILE)
        if values is None:
            return
        mask_threshold, boundary_radius = values
        blockers = [
            QSignalBlocker(self.polygon_compare_profile_combo),
        ]
        _ = blockers
        self.mask_threshold_spin.setValue(float(mask_threshold))
        self.boundary_radius_spin.setValue(int(boundary_radius))
        profile_index = self.polygon_compare_profile_combo.findData(profile)
        if profile_index < 0:
            profile_index = self.polygon_compare_profile_combo.findData(DEFAULT_POLYGON_COMPARE_PROFILE)
        self.polygon_compare_profile_combo.setCurrentIndex(profile_index if profile_index >= 0 else 0)

    def _append_folder_item(self, folder_path: Path, *, checked: bool) -> QListWidgetItem:
        folder_path = Path(folder_path)
        folder_path_text = str(folder_path)
        item_count = self.folder_list.count()
        for row in range(item_count):
            existing_item = self.folder_list.item(row)
            if str(existing_item.data(Qt.ItemDataRole.UserRole) or "") == folder_path_text:
                existing_item.setData(FOLDER_CHECKED_ROLE, bool(checked))
                if not existing_item.data(FOLDER_LABEL_ROLE):
                    existing_item.setData(FOLDER_LABEL_ROLE, folder_path.name)
                return existing_item
        item = QListWidgetItem()
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        item.setData(Qt.ItemDataRole.UserRole, folder_path_text)
        item.setData(FOLDER_CHECKED_ROLE, bool(checked))
        item.setData(FOLDER_LABEL_ROLE, folder_path.name)
        item.setData(FOLDER_CONFIDENCE_ROLE, "")
        item.setData(FOLDER_CONFIDENCE_EXPANDED_ROLE, False)
        item.setToolTip(folder_path_text)
        self.folder_list.addItem(item)
        return item

    @staticmethod
    def _folder_has_supported_images(folder_path: Path) -> bool:
        if not folder_path.exists() or not folder_path.is_dir():
            return False
        try:
            for image_path in folder_path.glob("*"):
                if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSION_SET:
                    return True
        except OSError as error:
            _LOGGER.warning("Could not inspect image folder %s: %s", folder_path, error)
            return False
        return False

    def _refresh_folder_rows(self) -> None:
        item_count = self.folder_list.count()
        self.folder_list.setUpdatesEnabled(False)
        try:
            for row in range(item_count):
                item = self.folder_list.item(row)
                path_text = str(item.data(Qt.ItemDataRole.UserRole))
                display_text = str(item.data(FOLDER_LABEL_ROLE) or (Path(path_text).name or path_text))
                confidence_path_text = str(item.data(FOLDER_CONFIDENCE_ROLE) or "")
                confidence_expanded = bool(item.data(FOLDER_CONFIDENCE_EXPANDED_ROLE))
                confidence_display_text = self._compact_path_text(confidence_path_text)
                row_widget = FolderRowWidget(
                    self.folder_list,
                    path_text=path_text,
                    display_text=display_text,
                    checked=bool(item.data(FOLDER_CHECKED_ROLE)),
                    confidence_display_text=confidence_display_text,
                    confidence_path_text=confidence_path_text,
                    confidence_expanded=confidence_expanded,
                    can_move_up=row > 0,
                    can_move_down=row < item_count - 1,
                    on_checked_changed=lambda checked, item=item: self._set_folder_item_checked(item, checked),
                    on_label_changed=lambda text, item=item: self._set_folder_item_label(item, text),
                    on_confidence_folder=lambda _checked=False, item=item: self._set_folder_item_confidence_folder(
                        item
                    ),
                    on_clear_confidence_folder=lambda _checked=False, item=item: (
                        self._clear_folder_item_confidence_folder(item)
                    ),
                    on_confidence_toggle=lambda expanded, item=item: self._set_folder_item_confidence_expanded(
                        item, expanded
                    ),
                    on_remove=lambda _checked=False, item=item: self._remove_folder_item(item),
                    on_move_up=lambda _checked=False, item=item: self._move_folder_item(item, -1),
                    on_move_down=lambda _checked=False, item=item: self._move_folder_item(item, 1),
                    checkbox_tooltip="Use model in analytics",
                    confidence_placeholder=self._t("folders.confidence_not_set"),
                    confidence_tooltip=confidence_path_text,
                    confidence_select_tooltip=self._t("folders.select_confidence"),
                    confidence_clear_tooltip=self._t("folders.clear_confidence"),
                    confidence_expand_tooltip=self._t("folders.show_confidence"),
                    confidence_collapse_tooltip=self._t("folders.hide_confidence"),
                    remove_tooltip="Remove model folder",
                    move_up_tooltip="Move up",
                    move_down_tooltip="Move down",
                )
                row_widget.setMinimumHeight(FOLDER_ROW_MIN_HEIGHT)
                item.setSizeHint(row_widget.sizeHint())
                self.folder_list.setItemWidget(item, row_widget)
        finally:
            self.folder_list.setUpdatesEnabled(True)

    def _set_folder_item_checked(self, item: QListWidgetItem, checked: bool) -> None:
        self._folder_check_guard = True
        item.setData(FOLDER_CHECKED_ROLE, bool(checked))
        self._folder_check_guard = False
        self._refresh_folder_rows()
        self._refresh_pair_matrix()
        self._sync_action_buttons()

    def _set_folder_item_label(self, item: QListWidgetItem, text: str) -> None:
        folder_path = Path(item.data(Qt.ItemDataRole.UserRole))
        item.setData(FOLDER_LABEL_ROLE, text or folder_path.name)
        self._refresh_folder_rows()
        self._refresh_pair_matrix()

    def _set_folder_item_confidence_folder(self, item: QListWidgetItem) -> None:
        if self._worker_thread is not None:
            return
        folder = QFileDialog.getExistingDirectory(self._view, self._t("dialog.select_model_confidence_folder"))
        if not folder:
            return
        folder_path = Path(folder)
        if not self._folder_has_supported_images(folder_path):
            QMessageBox.warning(
                self._view,
                self._t("dialog.warning_title"),
                f"Confidence folder has no supported images: {folder_path}",
            )
            return
        item.setData(FOLDER_CONFIDENCE_ROLE, str(folder_path))
        self._refresh_folder_rows()
        self._sync_action_buttons()

    def _clear_folder_item_confidence_folder(self, item: QListWidgetItem) -> None:
        item.setData(FOLDER_CONFIDENCE_ROLE, "")
        self._refresh_folder_rows()
        self._sync_action_buttons()

    def _set_folder_item_confidence_expanded(self, item: QListWidgetItem, expanded: bool) -> None:
        item.setData(FOLDER_CONFIDENCE_EXPANDED_ROLE, bool(expanded))
        self._refresh_folder_rows()

    def _remove_folder_item(self, item: QListWidgetItem) -> None:
        row = self.folder_list.row(item)
        if row < 0:
            return
        self.folder_list.takeItem(row)
        self._refresh_folder_rows()
        self._refresh_pair_matrix()
        self._sync_action_buttons()

    def _move_folder_item(self, item: QListWidgetItem, delta: int) -> None:
        row = self.folder_list.row(item)
        target_row = row + int(delta)
        if row < 0 or target_row < 0 or target_row >= self.folder_list.count():
            return
        moved_item = self.folder_list.takeItem(row)
        self.folder_list.insertItem(target_row, moved_item)
        self.folder_list.setCurrentRow(target_row)
        self._refresh_folder_rows()
        self._refresh_pair_matrix()

    def _build_layout_config(self) -> MatrixLayoutConfig:
        return MatrixLayoutConfig(
            mode="indexed_grid",
            total_frames=int(DEFAULT_TOTAL_FRAMES),
            frames_per_row=int(self.frames_per_row_spin.value()),
            rows=int(DEFAULT_MATRIX_ROWS),
            columns=int(DEFAULT_MATRIX_COLUMNS),
        )

    def _percentile_bin_bounds(self, bin_index: int) -> tuple[float, float]:
        normalized = max(0, min(int(bin_index), len(PERCENTILE_BAND_BOUNDS) - 1))
        low_bound, high_bound = PERCENTILE_BAND_BOUNDS[normalized]
        return float(low_bound), float(high_bound)

    def _records_in_percentile_bin(
        self, records: tuple[FrameRecord, ...] | list[FrameRecord], percentile_map: dict[str, float], bin_index: int
    ) -> tuple[FrameRecord, ...]:
        low_bound, high_bound = self._percentile_bin_bounds(bin_index)
        selected: list[FrameRecord] = []
        for record in records:
            if record.key not in percentile_map:
                continue
            percentile = float(percentile_map.get(record.key, 0.0))
            if bin_index >= len(PERCENTILE_BAND_BOUNDS) - 1:
                matches = low_bound <= percentile <= high_bound
            else:
                matches = low_bound <= percentile < high_bound
            if matches:
                selected.append(record)
        return tuple(selected)

    def _records_for_percentile_bin(
        self, state: ExtendMatrixTabState, metric_key: str, bin_index: int
    ) -> tuple[FrameRecord, ...]:
        available_keys = set(self._percentile_basis_keys_for_state(state, state.build_result))
        if str(metric_key) not in available_keys:
            return tuple()
        percentile_map = self._percentile_map_for_metric(state, str(metric_key))
        return self._records_in_percentile_bin(self._base_records_for_state(state), percentile_map, int(bin_index))

    @staticmethod
    def _selected_percentile_bin_for_metric(state: ExtendMatrixTabState, metric_key: str) -> int | None:
        if state.selected_percentile_metric_key == metric_key and state.selected_percentile_bin_index is not None:
            return int(state.selected_percentile_bin_index)
        if state.percentile_filter_metric_key == metric_key and state.percentile_filter_bin_index is not None:
            return int(state.percentile_filter_bin_index)
        return None

    @staticmethod
    def _safe_export_fragment(value: object, *, fallback: str) -> str:
        text = str(value or "").strip()
        text = re.sub(r"[^\w.\-]+", "_", text, flags=re.UNICODE).strip("._-")
        return text[:80] or str(fallback)

    @staticmethod
    def _percentile_filter_active(state: ExtendMatrixTabState) -> bool:
        return bool(state.percentile_filter_metric_key is not None and state.percentile_filter_bin_index is not None)

    def _percentile_filter_uses_full_matrix(self, state: ExtendMatrixTabState) -> bool:
        return bool(getattr(state, "percentile_filter_full_matrix", True))

    @staticmethod
    def _correlation_filter_active(state: ExtendMatrixTabState) -> bool:
        return str(getattr(state, "correlation_filter_band", "") or "") in {"bad", "good"}

    @staticmethod
    def _selected_correlation_band_for_state(state: ExtendMatrixTabState) -> str | None:
        selected = str(getattr(state, "selected_correlation_band", "") or "")
        if selected in {"bad", "good"}:
            return selected
        active = str(getattr(state, "correlation_filter_band", "") or "")
        return active if active in {"bad", "good"} else None

    def _max_correlation_limit_for_state(self, state: ExtendMatrixTabState) -> int:
        try:
            frame_count = int(self._percentile_base_record_count(state))
        except (AttributeError, TypeError, ValueError) as error:
            _LOGGER.debug("Could not derive percentile record count; using build records: %s", error)
            frame_count = int(len(getattr(state.build_result, "records", ()) or ()))
        return max(1, frame_count // 2)

    def _correlation_limit_for_state(self, state: ExtendMatrixTabState) -> int:
        max_limit = self._max_correlation_limit_for_state(state)
        spinbox = getattr(state, "correlation_limit_spin", None)
        if spinbox is not None:
            try:
                return max(1, min(max_limit, int(spinbox.value())))
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                _LOGGER.debug("Could not read correlation limit widget; using persisted value: %s", error)
        try:
            return max(1, min(max_limit, int(getattr(state, "correlation_limit", 25))))
        except (TypeError, ValueError):
            return min(max_limit, 25)

    def _sync_correlation_limit_bounds(self, state: ExtendMatrixTabState) -> int:
        max_limit = self._max_correlation_limit_for_state(state)
        value = self._correlation_limit_for_state(state)
        spinbox = getattr(state, "correlation_limit_spin", None)
        if spinbox is not None:
            blocker = QSignalBlocker(spinbox)
            spinbox.setRange(1, max_limit)
            spinbox.setValue(value)
            spinbox.setToolTip(self._t("hist.correlation_limit_hint", count=max_limit))
            del blocker
        state.correlation_limit = value
        return value

    def _percentile_highlight_keys_for_state(self, state: ExtendMatrixTabState) -> set[str]:
        if not self._percentile_filter_active(state) or not self._percentile_filter_uses_full_matrix(state):
            return set()
        percentile_map = self._percentile_map_for_metric(state, str(state.percentile_filter_metric_key))
        records = self._records_in_percentile_bin(
            self._base_records_for_state(state),
            percentile_map,
            int(state.percentile_filter_bin_index),
        )
        return {str(record.key) for record in records if str(record.key)}

    def _correlation_records_for_state(self, state: ExtendMatrixTabState, band: str) -> tuple[FrameRecord, ...]:
        if str(band) not in {"bad", "good"}:
            return tuple()
        limit = self._correlation_limit_for_state(state)
        return tuple(
            record for record, _count, _avg, _labels in self._repeated_percentile_entries(state, band=str(band))[:limit]
        )

    def _matrix_highlight_keys_for_state(self, state: ExtendMatrixTabState) -> set[str]:
        percentile_keys = self._percentile_highlight_keys_for_state(state)
        if self._correlation_filter_active(state) and self._percentile_filter_uses_full_matrix(state):
            return {
                str(record.key)
                for record in self._correlation_records_for_state(state, str(state.correlation_filter_band))
                if str(record.key)
            }
        return percentile_keys

    def _base_records_for_state(self, state: ExtendMatrixTabState) -> tuple[FrameRecord, ...]:
        cache_key = (
            str(getattr(state, "object_type", POLYGON_OBJECT_TYPE) or POLYGON_OBJECT_TYPE),
            id(state.build_result.records),
        )
        cached = state.base_records_cache.get(cache_key)
        if cached is not None:
            return tuple(cached)
        records: tuple[FrameRecord, ...] | list[FrameRecord] = state.build_result.records
        object_type = str(getattr(state, "object_type", POLYGON_OBJECT_TYPE) or POLYGON_OBJECT_TYPE)
        if object_type in {POLYGON_OBJECT_TYPE, POINT_OBJECT_TYPE}:
            records = tuple(
                record
                for record in records
                if record.summary is None
                or str(getattr(record.summary, "frame_type", POLYGON_OBJECT_TYPE)) == object_type
            )
        resolved = tuple(records)
        state.base_records_cache[cache_key] = resolved
        return resolved

    def _display_records_for_state(self, state: ExtendMatrixTabState) -> tuple[FrameRecord, ...]:
        records: tuple[FrameRecord, ...] | list[FrameRecord] = self._base_records_for_state(state)
        available_keys = set(self._percentile_basis_keys_for_state(state, state.build_result))
        if state.percentile_filter_metric_key not in available_keys:
            state.percentile_filter_metric_key = None
            state.percentile_filter_bin_index = None
        if state.selected_percentile_metric_key not in available_keys:
            state.selected_percentile_metric_key = None
            state.selected_percentile_bin_index = None
        if self._percentile_filter_active(state) and not self._percentile_filter_uses_full_matrix(state):
            percentile_map = self._percentile_map_for_metric(state, state.percentile_filter_metric_key)
            records = self._records_in_percentile_bin(records, percentile_map, state.percentile_filter_bin_index)
        if self._correlation_filter_active(state) and not self._percentile_filter_uses_full_matrix(state):
            allowed_keys = {
                record.key for record in self._correlation_records_for_state(state, str(state.correlation_filter_band))
            }
            records = tuple(record for record in records if record.key in allowed_keys)
        return tuple(records)

    def _capture_view_snapshot(self) -> dict[str, object]:
        confidence_model_id = self._selected_confidence_model_id(None)
        mask_threshold, boundary_radius = self._selected_polygon_compare_values()
        current_state = self._current_tab_state()
        return {
            "cell_size": int(DEFAULT_CELL_SIZE),
            "layout_config": self._build_layout_config(),
            "matrix_score_view_mode": str(self.matrix_score_view_combo.currentData() or DEFAULT_MATRIX_SCORE_VIEW_MODE),
            "gradient_name": str(self.matrix_gradient_combo.currentData() or DEFAULT_GRADIENT_NAME),
            "analysis_mode": self._selected_analysis_mode(),
            "comparison_target": self._selected_comparison_target().value,
            "object_type": self._selected_object_type(),
            "geometry_mode": str(self.geometry_mode_combo.currentData() or DEFAULT_GEOMETRY_MODE),
            "polygon_compare_profile": self._selected_polygon_compare_profile(),
            "mask_threshold": float(mask_threshold),
            "boundary_radius": int(boundary_radius),
            "confidence_uncertainty_profile": self._selected_confidence_uncertainty_profile(),
            "confidence_uncertainty_delta": self._selected_confidence_uncertainty_delta(),
            "point_match_radius": float(self.point_match_radius_spin.value()),
            "point_confidence_radius": int(self.point_confidence_radius_spin.value()),
            "point_extraction_mode": str(
                self.point_extraction_mode_combo.currentData() or DEFAULT_POINT_EXTRACTION_MODE
            ),
            "polygon_confidence_summary": str(
                self.polygon_confidence_summary_combo.currentData() or DEFAULT_POLYGON_CONFIDENCE_SUMMARY
            ),
            "metric_key": str(self.metric_combo.currentData() or DEFAULT_MATRIX_METRIC_KEY),
            "metric_scope": str(confidence_model_id or ""),
            "confidence_model_id": confidence_model_id,
            "frame_type_filter": str(self.frame_type_filter_combo.currentData() or "all"),
            "excluded_record_keys": tuple(sorted(self._excluded_record_keys_for_state(current_state))),
        }

    def _set_ui_context_from_state(self, state: ExtendMatrixTabState) -> None:
        score_view_blocker = QSignalBlocker(self.matrix_score_view_combo)
        score_view_index = self.matrix_score_view_combo.findData(
            str(state.matrix_score_view_mode or DEFAULT_MATRIX_SCORE_VIEW_MODE)
        )
        self.matrix_score_view_combo.setCurrentIndex(score_view_index if score_view_index >= 0 else 0)
        del score_view_blocker

        gradient_blocker = QSignalBlocker(self.matrix_gradient_combo)
        gradient_index = self.matrix_gradient_combo.findData(str(state.gradient_name or DEFAULT_GRADIENT_NAME))
        self.matrix_gradient_combo.setCurrentIndex(gradient_index if gradient_index >= 0 else 0)
        del gradient_blocker

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
        frame_type_filter_index = self.frame_type_filter_combo.findData(str(state.frame_type_filter or "all"))
        self.frame_type_filter_combo.setCurrentIndex(frame_type_filter_index if frame_type_filter_index >= 0 else 0)
        del frame_type_filter_blocker

        self._set_grid_inspection_config_controls(
            getattr(state, "grid_inspection_config_payload", {}) or self._grid_inspection_config_payload()
        )

    def _apply_global_analysis_context_to_state(
        self, state: ExtendMatrixTabState, *, update_frame_filter: bool = False
    ) -> bool:
        selected_analysis_mode = self._selected_analysis_mode()
        selected_object_type = self._selected_object_type()
        updated_options = self._analysis_options_from_controls(state, object_type=selected_object_type)
        context_changed = str(state.analysis_mode or "") != str(selected_analysis_mode) or str(
            state.object_type or ""
        ) != str(selected_object_type)
        options_changed = updated_options != state.build_result.options
        if context_changed or options_changed:
            state.build_result = replace(state.build_result, options=updated_options)
            self._invalidate_state_runtime_caches(state, clear_metric_results=True)
            state.last_analytics_request_signature = None
        else:
            state.build_result = replace(state.build_result, options=updated_options)

        state.analysis_mode = selected_analysis_mode
        state.object_type = selected_object_type
        if update_frame_filter:
            state.frame_type_filter = str(
                self.frame_type_filter_combo.currentData() or state.frame_type_filter or "all"
            )

        preferred_scope = str(
            self.metric_scope_combo.currentData() or state.confidence_model_id or state.metric_scope or ""
        )
        context = resolve_analysis_context(
            state.build_result,
            selected_analysis_mode,
            selected_object_type,
            confidence_model_id=preferred_scope,
        )
        state.confidence_model_id = str(context.confidence_model_id or "") or None
        state.metric_scope = str(state.confidence_model_id or "")
        return bool(context_changed or options_changed)

    def _apply_global_analysis_context_to_all_states(self) -> None:
        for state in tuple(self._tab_states.values()):
            self._apply_global_analysis_context_to_state(state)

    def _analysis_options_from_controls(self, state: ExtendMatrixTabState, *, object_type: str) -> BuildOptions:
        return replace(
            state.build_result.options,
            geometry_mode=geometry_mode_for_object_type(object_type),
            mask_threshold=float(self.mask_threshold_spin.value()),
            boundary_radius=int(self.boundary_radius_spin.value()),
            confidence_uncertainty_delta=self._selected_confidence_uncertainty_delta(),
            point_match_radius=float(self.point_match_radius_spin.value()),
            point_confidence_radius=int(self.point_confidence_radius_spin.value()),
            point_extraction_mode=str(self.point_extraction_mode_combo.currentData() or DEFAULT_POINT_EXTRACTION_MODE),
            polygon_confidence_summary=str(
                self.polygon_confidence_summary_combo.currentData() or DEFAULT_POLYGON_CONFIDENCE_SUMMARY
            ),
            comparison_pairs=self._selected_comparison_pairs(),
            comparison_target=self._selected_comparison_target(),
        )

    def _invalidate_state_runtime_caches(
        self, state: ExtendMatrixTabState, *, clear_metric_results: bool = False
    ) -> None:
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

    def _analytics_request_signature(
        self, state: ExtendMatrixTabState, metric_key: str | None = None
    ) -> tuple[object, ...]:
        return (
            "analytics",
            id(state.widget),
            str(state.analysis_mode),
            str(state.object_type),
            str(metric_key or state.metric_key or DEFAULT_MATRIX_METRIC_KEY),
            str(state.confidence_model_id or ""),
            tuple(sorted(self._excluded_record_keys_for_state(state))),
            state.build_result.options,
        )

    def _update_processing_visuals(self, state: ExtendMatrixTabState | None) -> None:
        if state is None:
            return
        state.matrix_view.set_processing_keys(set(self._active_processing_keys))
        if self._worker_kind == "grid_inspection" and hasattr(self, "grid_inspection_matrix_view"):
            for view in self._grid_inspection_views().values():
                view.set_processing_keys(set(self._active_processing_keys))

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
            self.build_progress.setFormat(
                self._progress_format_text(
                    self._active_progress_current, self._active_progress_total, self._active_progress_key
                )
            )

    def _on_frame_states_changed(self, keys: object, status: str, *, generation: int | None = None) -> None:
        if not self._is_active_request_generation(generation):
            return
        state = self._active_compute_state
        if state is None:
            return
        normalized_status = str(status or "running").lower()
        normalized_keys = (
            {str(key) for key in (keys or ()) if str(key)} if isinstance(keys, (list, tuple, set)) else set()
        )
        if not normalized_keys:
            return
        for key in normalized_keys:
            state.processing_state_by_key[key] = normalized_status
        if normalized_status == "running":
            self._active_processing_keys.update(normalized_keys)
        else:
            self._active_processing_keys.difference_update(normalized_keys)
        self._update_processing_visuals(state)

    def _sync_current_analysis_context(self, state: ExtendMatrixTabState, *, auto_recompute: bool) -> None:
        options_changed = self._apply_global_analysis_context_to_state(state, update_frame_filter=True)

        self._sync_metric_controls(
            state.build_result,
            preferred_metric_key=state.metric_key,
            preferred_scope_key=state.confidence_model_id or state.metric_scope,
            context_state=state,
        )
        metric_key = str(
            self.metric_combo.currentData() or self._default_metric_key_for_state(state, state.build_result)
        )
        state.metric_key = metric_key

        if self._worker_thread is not None:
            return
        requires_analytics = options_changed or self._metric_value_missing_for_build_result(
            state.build_result, metric_key
        )
        if requires_analytics:
            if auto_recompute and bool(getattr(state.build_result, "scores_computed", False)):
                self._start_compute_analytics(state=state, sync_context=False)
            return
        self._apply_metric_to_state(state, metric_key)

    def _metric_scope_for_metric_key(self, metric_key: str) -> str:
        family = confidence_metric_family(metric_key)
        if family is None:
            return ""
        _metric_family, model_id = family
        return str(model_id)

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
        selected_confidence_model_id = str(
            preferred_scope_key
            or self.metric_scope_combo.currentData()
            or self._metric_scope_for_metric_key(metric_key)
            or ""
        )
        context = self._analysis_context_for_state(state, build_result)
        self._populate_metric_scope_combo(
            build_result,
            selected_confidence_model_id,
            output_only=context.analysis_mode == MODEL_OUTPUT_CONFIDENCE_MODE,
        )
        if build_result is not None and self._confidence_context_available(context, build_result):
            selected_confidence_model_id = str(
                self.metric_scope_combo.currentData()
                or default_confidence_model_id(
                    build_result, output_only=context.analysis_mode == MODEL_OUTPUT_CONFIDENCE_MODE
                )
                or ""
            )
            context = resolve_analysis_context(
                build_result,
                context.analysis_mode,
                context.object_type,
                confidence_model_id=selected_confidence_model_id,
            )
        available = set(getattr(build_result, "available_metric_keys", ()) or ())
        basis_keys = [str(key) for key in percentile_basis_keys(context) if build_result is None or key in available]
        if build_result is not None and self._is_dynamic_pair_metric_key(metric_key) and metric_key not in basis_keys:
            basis_keys.append(metric_key)
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
            self.metric_combo.setItemData(
                combo_index, self._metric_hint_fallback(key, build_result), Qt.ItemDataRole.ToolTipRole
            )
        metric_index = self.metric_combo.findData(metric_key)
        self.metric_combo.setCurrentIndex(metric_index if metric_index >= 0 else 0)
        self.metric_combo.setToolTip(self._metric_hint_fallback(metric_key, build_result))
        self.metric_combo.blockSignals(False)

    def _attach_matrix_coordinates(
        self, records: tuple[FrameRecord, ...] | list[FrameRecord], layout_config: MatrixLayoutConfig
    ) -> tuple[FrameRecord, ...]:
        records_tuple = tuple(records)
        placements, _columns, _rows = build_matrix_layout(list(records_tuple), layout_config)
        records_by_key: dict[str, FrameRecord] = {}
        changed = False
        for placement_index, (record, row, column) in enumerate(placements):
            identity = record.identity
            if identity is None:
                identity = FrameIdentity(
                    frame_id=placement_index, base_id=placement_index, tile_x=column, tile_y=row, source_key=record.key
                )
                records_by_key[record.key] = replace(record, identity=identity)
                changed = True
                continue
            if int(identity.tile_x if identity.tile_x is not None else -1) == int(column) and int(
                identity.tile_y if identity.tile_y is not None else -1
            ) == int(row):
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
        state.gradient_name = str(self.matrix_gradient_combo.currentData() or DEFAULT_GRADIENT_NAME)
        state.frame_type_filter = str(self.frame_type_filter_combo.currentData() or "all")

    def _apply_tab_visual_settings(
        self, state: ExtendMatrixTabState, *, reset_view: bool = False, update_histograms: bool = True
    ) -> bool:
        try:
            self._sync_state_record_coordinates(state)
            display_records = self._display_records_for_state(state)
            state.matrix_view.set_gradient_preset(state.gradient_name or DEFAULT_GRADIENT_NAME)
            state.matrix_view.set_cell_size(int(state.cell_size))
            state.matrix_view.set_layout_config(state.layout_config)
            state.matrix_view.set_score_view_mode(str(state.matrix_score_view_mode or DEFAULT_MATRIX_SCORE_VIEW_MODE))
            state.matrix_view.set_metric_context(
                state.metric_key,
                point_match_radius=float(getattr(state.build_result.options, "point_match_radius", 3.0)),
                bce_score_cap=float(BCE_SCORE_CAP),
            )
            state.matrix_view.set_reference_key(state.build_result.best_match_key)
            sort_mode = "input_order" if str(state.layout_config.mode or "indexed_grid") == "manual_grid" else "name"
            percentile_filters_records = self._percentile_filter_active(
                state
            ) and not self._percentile_filter_uses_full_matrix(state)
            correlation_filters_records = self._correlation_filter_active(
                state
            ) and not self._percentile_filter_uses_full_matrix(state)
            filtered_view = percentile_filters_records or correlation_filters_records
            state.matrix_view.set_highlighted_record_keys(self._matrix_highlight_keys_for_state(state))
            state.matrix_view.set_records(
                list(display_records),
                sort_mode=sort_mode,
                reset_view=reset_view,
                prefer_complete=filtered_view,
            )
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
        if str(metric_key or "") in {
            "confidence_model_score",
            "confidence_difference_score",
            "confidence_bce_score",
            "confidence_threshold_crossing_score",
        }:
            for record in build_result.records:
                summary = record.summary
                if summary is None:
                    continue
                if metric_key in getattr(summary, "metric_values", {}):
                    return False
            return True
        if parse_pair_metric_key(metric_key) is not None:
            for record in build_result.records:
                summary = record.summary
                if summary is None:
                    continue
                if metric_key in getattr(summary, "metric_values", {}):
                    return False
            return True
        if (
            parse_confidence_pair_metric_key(metric_key) is not None
            or parse_combined_pair_metric_key(metric_key) is not None
        ):
            for record in build_result.records:
                summary = record.summary
                if summary is None:
                    continue
                if metric_key in getattr(summary, "metric_values", {}):
                    return False
            return True
        parsed = _parse_model_metric_key(metric_key)
        if parsed is None:
            return False
        family, model_id = parsed
        if family == "model_output_confidence" and not any(
            bool((record.model_prob_paths or {}).get(model_id)) for record in build_result.records
        ):
            return False
        if family == "model_uncertain_fraction":
            frame_type = next(
                (
                    str(record.summary.frame_type)
                    for record in build_result.records
                    if record.summary is not None and getattr(record.summary, "frame_type", None)
                ),
                None,
            )
            if frame_type == "point":
                return False
        if family == "model_point_contrast":
            frame_type = next(
                (
                    str(record.summary.frame_type)
                    for record in build_result.records
                    if record.summary is not None and getattr(record.summary, "frame_type", None)
                ),
                None,
            )
            if frame_type == "polygon":
                return False
        for record in build_result.records:
            summary = record.summary
            if summary is None:
                continue
            if metric_key in getattr(summary, "metric_values", {}):
                return False
        return True

    def _metric_higher_is_better(self, metric_key: str) -> bool:
        if str(metric_key) == GRID_INSPECTION_DAMAGE_METRIC_KEY:
            return False
        return metric_higher_is_better(metric_key)

    def _metric_score_style(self, value: float | None, metric_key: str) -> str:
        ratio = metric_visual_ratio(
            metric_key,
            value,
            point_match_radius=float(self.point_match_radius_spin.value()),
            bce_score_cap=float(BCE_SCORE_CAP),
        )
        level_key = metric_level_key(
            metric_key,
            value,
            point_match_radius=float(self.point_match_radius_spin.value()),
            bce_score_cap=float(BCE_SCORE_CAP),
        )
        higher_is_better = self._metric_higher_is_better(metric_key)
        family = str(metric_key or "").split("::", 1)[0]
        if ratio is None or level_key is None:
            background = "#2f3844"
            foreground = "#edf3fb"
        elif family in {"model_confidence", "model_output_confidence"}:
            if level_key == "score.level.low":
                background = "#1f5f3b"
                foreground = "#e9fff1"
            elif level_key == "score.level.moderate":
                background = "#6f7a18"
                foreground = "#f7ffd8"
            elif level_key == "score.level.elevated":
                background = "#a75d12"
                foreground = "#fff0dc"
            else:
                background = "#8c2f39"
                foreground = "#ffe9ec"
        elif higher_is_better:
            if ratio < 0.33:
                background = "#8c2f39"
                foreground = "#ffe9ec"
            elif ratio < 0.66:
                background = "#8a6a12"
                foreground = "#fff7da"
            else:
                background = "#1f5f3b"
                foreground = "#e9fff1"
        else:
            if ratio < 0.33:
                background = "#1f5f3b"
                foreground = "#e9fff1"
            elif ratio < 0.66:
                background = "#8a6a12"
                foreground = "#fff7da"
            else:
                background = "#8c2f39"
                foreground = "#ffe9ec"
        return f"padding: 6px 10px; border-radius: 8px; background-color: {background}; color: {foreground}; font-weight: 700;"

    def _metric_score_text(self, value: float | None, metric_key: str) -> str:
        if value is None:
            return "-"
        level_key = metric_level_key(
            metric_key,
            value,
            point_match_radius=float(self.point_match_radius_spin.value()),
            bce_score_cap=float(BCE_SCORE_CAP),
        )
        if level_key is None:
            return "-"
        level = self._t(level_key)
        if "::" in str(metric_key):
            return f"{level} {float(value) * 100.0:.1f}%"
        if str(metric_key) in {
            "overall_polygon_score",
            "iou_score",
            "dice_score",
            "polygon_bce_score",
            "overall_point_score",
            "precision_score",
            "recall_score",
            "f1_score",
            "localization_score",
            "confidence_model_score",
            "confidence_difference_score",
            "confidence_bce_score",
            "confidence_threshold_crossing_score",
        }:
            return f"{level} {float(value):.1f}"
        return f"{level} {float(value):.4f}"

    def _metric_label(self, metric_key: str, build_result: BuildResult | None = None) -> str:
        metric_key_text = str(metric_key)
        if metric_key_text == GRID_INSPECTION_DAMAGE_METRIC_KEY:
            return self._t("metric.grid_inspection_damage_score")
        for label_key, key, _group in MATRIX_METRIC_OPTIONS:
            if str(key) == metric_key_text:
                return self._t(str(label_key))
        parsed_pair = parse_pair_metric_key(metric_key_text)
        if parsed_pair is not None:
            model_a, model_b, operation = parsed_pair
            display_names = {
                str(spec.model_id): str(spec.display_name or spec.model_id)
                for spec in tuple(getattr(build_result, "model_specs", ()) or ())
            }
            left = display_names.get(model_a, self._pair_display_name(model_a))
            right = display_names.get(model_b, self._pair_display_name(model_b))
            return f"{left} -> {right} [{PAIR_OPERATION_LABELS.get(operation, operation)}]"
        parsed_confidence_pair = parse_confidence_pair_metric_key(metric_key_text)
        if parsed_confidence_pair is not None:
            model_a, model_b, operation = parsed_confidence_pair
            display_names = {
                str(spec.model_id): str(spec.display_name or spec.model_id)
                for spec in tuple(getattr(build_result, "model_specs", ()) or ())
            }
            left = display_names.get(model_a, self._pair_display_name(model_a))
            right = display_names.get(model_b, self._pair_display_name(model_b))
            operation_label = {
                "mae": "MAE",
                "rmse": "RMSE",
                "mean_delta": "mean delta",
                "correlation": "correlation",
                "low_iou": "low-confidence IoU",
                "disagreement": "confidence disagreement",
            }.get(operation, operation)
            return f"{left} -> {right} [{operation_label}]"
        parsed_combined_pair = parse_combined_pair_metric_key(metric_key_text)
        if parsed_combined_pair is not None:
            model_a, model_b, _operation = parsed_combined_pair
            display_names = {
                str(spec.model_id): str(spec.display_name or spec.model_id)
                for spec in tuple(getattr(build_result, "model_specs", ()) or ())
            }
            left = display_names.get(model_a, self._pair_display_name(model_a))
            right = display_names.get(model_b, self._pair_display_name(model_b))
            return f"{left} -> {right} [combined risk]"
        translated = self._t(f"metric.{metric_key_text}")
        if translated != f"metric.{metric_key_text}":
            return translated
        if "::" in metric_key_text:
            family, model_id = metric_key_text.split("::", 1)
            model_name = model_id
            if build_result is not None:
                for spec in build_result.model_specs:
                    if spec.model_id == model_id:
                        model_name = spec.display_name
                        break
            if family == "model_confidence":
                return f"{self._t('metric.model_confidence')} [{model_name}]"
            if family == "model_output_confidence":
                return f"{self._t('metric.model_output_confidence')} [{model_name}]"
            if family == "model_uncertain_fraction":
                return f"{self._t('metric.model_uncertain_fraction')} [{model_name}]"
            if family == "model_point_contrast":
                return f"{self._t('metric.model_point_contrast')} [{model_name}]"
        return metric_key_text

    def _metric_hint(self, metric_key: str, summary) -> str | None:
        metric_key_text = str(metric_key)
        if metric_key_text == GRID_INSPECTION_DAMAGE_METRIC_KEY:
            return self._t("hint.grid_inspection_percentiles")
        if (
            parse_pair_metric_key(metric_key_text) is not None
            or parse_confidence_pair_metric_key(metric_key_text) is not None
            or parse_combined_pair_metric_key(metric_key_text) is not None
        ):
            return (
                self._t("hint.inter_model_polygon")
                if summary.frame_type != "point"
                else self._t("hint.inter_model_point")
            )
        if "::" in metric_key_text:
            family, _model_id = metric_key_text.split("::", 1)
            if family == "model_output_confidence":
                return self._t("hint.model_output_confidence")
            if family in {"model_confidence", "model_uncertain_fraction", "model_point_contrast"}:
                if summary.frame_type == "point":
                    return self._t("hint.intra_model_point")
                return self._t("hint.confidence_polygon")
        if metric_key_text == "overall_frame_score":
            return self._t("hint.overall")
        if metric_key_text in {
            "overall_polygon_score",
            "iou_score",
            "dice_score",
            "polygon_bce_score",
            "iou",
            "dice",
            "bce",
        }:
            return self._t("hint.inter_model_polygon")
        if metric_key_text in {
            "overall_point_score",
            "precision_score",
            "recall_score",
            "f1_score",
            "localization_score",
            "precision",
            "recall",
            "f1",
            "mean_localization_distance",
        }:
            return self._t("hint.inter_model_point")
        if metric_key_text in {
            "confidence_model_score",
            "confidence_difference_score",
            "confidence_bce_score",
            "confidence_threshold_crossing_score",
        }:
            return self._t("hint.confidence_comparison")
        if metric_key_text == "model_model_score":
            return (
                self._t("hint.model_model_point")
                if summary.frame_type == "point"
                else self._t("hint.model_model_polygon")
            )
        return None

    def _metric_hint_fallback(self, metric_key: str, build_result: BuildResult | None = None) -> str:
        metric_key_text = str(metric_key)
        sample_summary = None
        if build_result is not None:
            for record in build_result.records:
                if record.summary is not None:
                    sample_summary = record.summary
                    break
        if sample_summary is not None:
            hint = self._metric_hint(metric_key_text, sample_summary)
            if hint:
                return hint
        family = metric_key_text.split("::", 1)[0]
        if self._is_dynamic_pair_metric_key(metric_key_text):
            return self._metric_label(metric_key_text, build_result)
        defaults = {
            GRID_INSPECTION_DAMAGE_METRIC_KEY: self._t("hint.grid_inspection_percentiles"),
            "overall_frame_score": self._t("hint.overall"),
            "model_model_score": self._t("hint.model_model_polygon"),
            "model_confidence": self._t("hint.confidence_polygon"),
            "model_output_confidence": self._t("hint.model_output_confidence"),
            "model_uncertain_fraction": self._t("hint.confidence_polygon"),
            "model_point_contrast": self._t("hint.confidence_point"),
            "overall_polygon_score": self._t("hint.inter_model_polygon"),
            "iou_score": self._t("hint.inter_model_polygon"),
            "dice_score": self._t("hint.inter_model_polygon"),
            "polygon_bce_score": self._t("hint.inter_model_polygon"),
            "iou": self._t("hint.inter_model_polygon"),
            "dice": self._t("hint.inter_model_polygon"),
            "bce": self._t("hint.inter_model_polygon"),
            "overall_point_score": self._t("hint.inter_model_point"),
            "precision_score": self._t("hint.inter_model_point"),
            "recall_score": self._t("hint.inter_model_point"),
            "f1_score": self._t("hint.inter_model_point"),
            "localization_score": self._t("hint.inter_model_point"),
            "precision": self._t("hint.inter_model_point"),
            "recall": self._t("hint.inter_model_point"),
            "f1": self._t("hint.inter_model_point"),
            "mean_localization_distance": self._t("hint.inter_model_point"),
            "confidence_model_score": self._t("hint.confidence_comparison"),
            "confidence_difference_score": self._t("hint.confidence_comparison"),
            "confidence_bce_score": self._t("hint.confidence_comparison"),
            "confidence_threshold_crossing_score": self._t("hint.confidence_comparison"),
        }
        return defaults.get(family, self._metric_label(metric_key_text, build_result))

    def _metric_component_summary(self, metric_key: str, summary) -> str:
        metric_key_text = str(metric_key)
        family = metric_key_text.split("::", 1)[0]
        is_ru = getattr(self._i18n, "language", "en") == "ru"
        if family == "overall_frame_score":
            return self._t("metric.disagreement_score")
        if family in {"overall_polygon_score", "iou_score", "dice_score", "polygon_bce_score", "iou", "dice", "bce"}:
            return "iou + dice + bce score"
        if parse_pair_metric_key(metric_key_text) is not None:
            return "pair XOR + IoU + Dice"
        if parse_confidence_pair_metric_key(metric_key_text) is not None:
            return "confidence MAE + RMSE + correlation + low-confidence overlap"
        if parse_combined_pair_metric_key(metric_key_text) is not None:
            return "output disagreement + confidence disagreement"
        if family in {
            "confidence_model_score",
            "confidence_difference_score",
            "confidence_bce_score",
            "confidence_threshold_crossing_score",
        }:
            return "confidence difference + confidence BCE + threshold crossing"
        if family in {
            "overall_point_score",
            "precision_score",
            "recall_score",
            "f1_score",
            "localization_score",
            "precision",
            "recall",
            "f1",
            "mean_localization_distance",
        }:
            return "precision + recall + f1 + localization + tp/fp/fn"
        if family == "model_model_score":
            return (
                "soft_dice + soft_iou + ssim + dice + iou + hausdorff + centroid"
                if summary.frame_type != "point"
                else (
                    "precision + recall + f1_at_r + локализация + количество"
                    if is_ru
                    else "precision + recall + f1_at_r + localization + count"
                )
            )
        if family == "model_confidence":
            return (
                "средняя уверенность внутри объекта"
                if (is_ru and summary.frame_type != "point")
                else (
                    "средняя уверенность по точкам"
                    if is_ru
                    else ("mean object confidence" if summary.frame_type != "point" else "mean point confidence")
                )
            )
        if family == "model_output_confidence":
            return "неуверенность confidence-выхода модели" if is_ru else "model confidence-output uncertainty"
        if family == "model_uncertain_fraction":
            return "доля сомнительных пикселей объекта" if is_ru else "uncertain object fraction"
        if family == "model_point_contrast":
            return "средний контраст точек" if is_ru else "mean point contrast"
        if family == "disagreement_score":
            return "1 - согласие моделей" if is_ru else "1 - model-to-model agreement"
        return "-"

    def _component_name_label(self, name: str) -> str:
        labels_en = {
            "source": "Source",
            "formula": "Formula",
            "definition": "Definition",
            "value": "Value",
            "model": "Model",
            "frame_uncertainty_score": "Frame uncertainty score",
            "mean_uncertainty": "Mean uncertainty",
            "low_conf_fraction": "Low-confidence fraction",
            "worst_tail_uncertainty": "Worst-tail uncertainty",
            "largest_low_conf_component": "Largest low-confidence component",
            "uncertain_support_fraction": "Uncertain support fraction",
            "top_uncertainty_mean": "Top uncertainty mean",
            "largest_uncertain_region_fraction": "Largest uncertain region fraction",
            "mean_object_confidence": "Mean object confidence",
            "mean_object_probability": "Mean object probability",
            "uncertain_fraction": "Uncertain fraction",
            "object_area_fraction": "Object area fraction",
            "mean_point_confidence": "Mean point confidence",
            "mean_point_probability": "Mean point probability",
            "mean_point_contrast": "Mean point contrast",
            "point_count": "Point count",
            "soft_dice": "Soft Dice",
            "soft_iou": "Soft IoU",
            "ssim": "SSIM",
            "dice": "Dice",
            "iou": "IoU",
            "hausdorff_distance": "Hausdorff distance",
            "centroid_distance": "Centroid distance",
            "mae": "MAE",
            "rmse": "RMSE",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "f1_at_r": "F1@r",
            "tp": "TP",
            "fp": "FP",
            "fn": "FN",
            "bce": "BCE",
            "iou_score": "IoU score",
            "dice_score": "Dice score",
            "polygon_bce_score": "BCE score",
            "overall_polygon_score": "Overall polygon score",
            "precision_score": "Precision score",
            "recall_score": "Recall score",
            "f1_score": "F1 score",
            "overall_point_score": "Overall point score",
            "mean_localization_distance": "Mean localization distance",
            "mean_localization_error": "Mean localization error",
            "localization_score": "Localization score",
            "localization_agreement": "Localization agreement",
            "count_error": "Count error",
            "count_agreement": "Count agreement",
            "connected_component_error": "Connected-component error",
            "cc_error": "Connected-component error",
            "chamfer_score": "Chamfer score",
            "hausdorff_score": "Hausdorff score",
        }
        labels_ru = {
            "source": "Источник",
            "formula": "Формула",
            "definition": "Определение",
            "value": "Значение",
            "model": "Модель",
            "hot_region_count": "Число горячих областей",
            "acquisition_score": "Приоритет на разметку",
            "mean_object_confidence": "Средняя уверенность внутри объекта",
            "mean_object_probability": "Среднее grayscale-значение внутри объекта",
            "uncertain_fraction": "Доля сомнительных пикселей",
            "object_area_fraction": "Доля площади объекта",
            "mean_point_confidence": "Средняя уверенность по точкам",
            "mean_point_probability": "Среднее grayscale-значение по точкам",
            "mean_point_contrast": "Средний контраст точек",
            "point_count": "Количество точек",
            "soft_dice": "Soft Dice",
            "soft_iou": "Soft IoU",
            "ssim": "SSIM",
            "dice": "Dice",
            "iou": "IoU",
            "hausdorff_distance": "Расстояние Хаусдорфа",
            "centroid_distance": "Расстояние между центроидами",
            "mae": "MAE",
            "rmse": "RMSE",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "f1_at_r": "F1@r",
            "tp": "TP",
            "fp": "FP",
            "fn": "FN",
            "bce": "BCE",
            "iou_score": "Score IoU",
            "dice_score": "Score Dice",
            "polygon_bce_score": "Score BCE",
            "overall_polygon_score": "Итоговый score полигонов",
            "precision_score": "Score Precision",
            "recall_score": "Score Recall",
            "f1_score": "Score F1",
            "overall_point_score": "Итоговый score точек",
            "mean_localization_distance": "Средняя ошибка локализации",
            "mean_localization_error": "Средняя ошибка локализации",
            "localization_score": "Оценка локализации",
            "localization_agreement": "Согласованность локализации",
            "count_error": "Ошибка количества",
            "count_agreement": "Согласованность количества",
            "connected_component_error": "Ошибка числа компонент",
            "cc_error": "Ошибка числа компонент",
            "chamfer_score": "Оценка Chamfer",
            "hausdorff_score": "Оценка Хаусдорфа",
        }
        labels = labels_ru if getattr(self._i18n, "language", "en") == "ru" else labels_en
        if name in labels:
            return labels[name]
        return name.replace("_", " ")

    def _component_value_text(self, value: str) -> str:
        values_en = {}
        values_ru = {
            "variance/entropy risk map": "карта риска по variance / entropy",
            "mean entropy of consensus probability": "средняя энтропия consensus probability",
            "mean variance over model probability maps": "средняя дисперсия probability maps моделей",
            "acquisition_score": "приоритет на разметку",
        }
        values = values_ru if getattr(self._i18n, "language", "en") == "ru" else values_en
        return values.get(
            value, value.replace(" vs ", " против ") if getattr(self._i18n, "language", "en") == "ru" else value
        )

    def _decorate_metric_lines(self, metric_key: str, summary, lines: list[str]) -> list[str]:
        decorated: list[str] = []
        hint = self._metric_hint(metric_key, summary)
        if hint:
            decorated.append(hint)
        for line in lines:
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            status_prefix = ""
            for status in ("active", "auxiliary", "legacy"):
                prefix = f"{status} "
                if stripped.startswith(prefix):
                    status_prefix = f"{self._t(f'status.{status}')} "
                    stripped = stripped[len(prefix) :]
                    break
            if ":" in stripped:
                name, value = stripped.split(":", 1)
                name = self._component_name_label(name.strip())
                value = self._component_value_text(value.strip())
                stripped = f"{status_prefix}{name}: {value}"
            else:
                stripped = status_prefix + self._component_value_text(stripped)
            decorated.append(indent + stripped)
        return decorated

    def _percentile_style(self, percentile: float | None, metric_key: str | None = None) -> str:
        if percentile is None:
            return self._metric_score_style(None, "overall_polygon_score")
        clipped = max(0.0, min(float(percentile), 100.0))
        # Percentiles shown in the UI are always goodness percentiles:
        # low percentile means a worse frame, high percentile means a better one.
        if clipped < 15.0:
            background = "#8c2f39"
            foreground = "#ffe9ec"
        elif clipped < 35.0:
            background = "#a75d12"
            foreground = "#fff0dc"
        elif clipped < 60.0:
            background = "#6f7a18"
            foreground = "#f7ffd8"
        else:
            background = "#1f5f3b"
            foreground = "#e9fff1"
        return f"padding: 6px 10px; border-radius: 8px; background-color: {background}; color: {foreground}; font-weight: 700;"

    def _percentile_text(self, percentile: float | None) -> str:
        if percentile is None:
            return "-"
        return f"P{float(percentile):.1f}"

    def _grid_inspection_percentile_map_for_state(self, state: ExtendMatrixTabState) -> dict[str, float]:
        excluded_keys = self._excluded_record_keys_for_state(state)
        payloads = dict(getattr(state, "grid_inspection_payload_by_key", {}) or {})
        cache_key = (
            GRID_INSPECTION_DAMAGE_METRIC_KEY,
            id(getattr(state, "grid_inspection_payload_by_key", None)),
            tuple(sorted(excluded_keys)),
        )
        cached = state.percentile_cache.get(cache_key)
        if cached is not None:
            return cached
        scored: list[tuple[float, str]] = []
        for record in getattr(state.build_result, "records", ()) or ():
            key = str(record.key)
            if key in excluded_keys:
                continue
            result = payloads.get(key)
            if result is None:
                continue
            try:
                score = float(getattr(result, "damage_score", getattr(result, "score", 0.0)) or 0.0)
            except (TypeError, ValueError) as error:
                _LOGGER.warning("Ignoring invalid grid damage score for %s: %s", key, error)
                continue
            if not isfinite(score):
                continue
            scored.append((max(0.0, min(1.0, score)), key))
        ranked_keys = [key for _score, key in sorted(scored, key=lambda item: (item[0], item[1]))]
        if not ranked_keys:
            percentile_map: dict[str, float] = {}
        elif len(ranked_keys) == 1:
            percentile_map = {ranked_keys[0]: 100.0}
        else:
            denominator = max(1, len(ranked_keys) - 1)
            percentile_map = {
                key: float(100.0 * (denominator - index) / denominator) for index, key in enumerate(ranked_keys)
            }
        state.percentile_cache[cache_key] = percentile_map
        return percentile_map

    def _percentile_map_for_metric(self, state: ExtendMatrixTabState, metric_key: str) -> dict[str, float]:
        if str(metric_key) == GRID_INSPECTION_DAMAGE_METRIC_KEY:
            return self._grid_inspection_percentile_map_for_state(state)
        base_records = self._base_records_for_state(state)
        cache_key = (str(metric_key), id(base_records))
        cached = state.percentile_cache.get(cache_key)
        if cached is not None:
            return cached
        percentile_map = compute_metric_percentiles(base_records, metric_key)
        state.percentile_cache[cache_key] = percentile_map
        return percentile_map

    def _percentile_histogram_counts(self, state: ExtendMatrixTabState, metric_key: str) -> list[int]:
        percentiles = self._percentile_map_for_metric(state, metric_key)
        counts = [0] * len(PERCENTILE_BAND_BOUNDS)
        upper_bounds = [float(high) for _low, high in PERCENTILE_BAND_BOUNDS[:-1]]
        last_index = len(counts) - 1
        for value in percentiles.values():
            clipped = max(0.0, min(float(value), 100.0))
            counts[min(last_index, bisect_right(upper_bounds, clipped))] += 1
        return counts

    def _repeated_percentile_entries(
        self, state: ExtendMatrixTabState, *, band: str
    ) -> list[tuple[FrameRecord, int, float, list[str]]]:
        available_keys = list(self._percentile_basis_keys_for_state(state, state.build_result))
        source_identity = (
            id(getattr(state, "grid_inspection_payload_by_key", None))
            if self._current_app_mode() == "grid_inspection"
            else id(state.build_result.records)
        )
        cache_key = (str(band), tuple(str(key) for key in available_keys), source_identity)
        cached = state.repeated_percentile_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        metrics_by_record: dict[str, list[tuple[str, float]]] = {
            record.key: [] for record in state.build_result.records
        }
        for metric_key in available_keys:
            percentile_map = self._percentile_map_for_metric(state, metric_key)
            for record in state.build_result.records:
                percentile = percentile_map.get(record.key)
                if percentile is None:
                    continue
                if band == "bad" and float(percentile) < 15.0:
                    metrics_by_record[record.key].append(
                        (self._metric_label(metric_key, state.build_result), float(percentile))
                    )
                elif band == "good" and float(percentile) >= 60.0:
                    metrics_by_record[record.key].append(
                        (self._metric_label(metric_key, state.build_result), float(percentile))
                    )
        entries: list[tuple[FrameRecord, int, float, list[str]]] = []
        records_by_key = {record.key: record for record in state.build_result.records}
        for key, values in metrics_by_record.items():
            if not values:
                continue
            metric_labels = [label for label, _percentile in values]
            average_percentile = sum(percentile for _label, percentile in values) / float(len(values))
            record = records_by_key[key]
            entries.append((record, len(values), average_percentile, metric_labels))
        if band == "bad":
            entries.sort(key=lambda item: (-item[1], item[2], item[0].display_name.lower()))
        else:
            entries.sort(key=lambda item: (-item[1], -item[2], item[0].display_name.lower()))
        state.repeated_percentile_cache[cache_key] = tuple(entries)
        return entries

    def _percentile_widgets_for_state(
        self, state: ExtendMatrixTabState
    ) -> tuple[dict[str, object], object | None, object | None]:
        if self._current_app_mode() == "grid_inspection":
            return (
                dict(getattr(self, "grid_inspection_histogram_cards", {}) or {}),
                getattr(self, "grid_inspection_repeated_bad_column", None),
                getattr(self, "grid_inspection_repeated_good_column", None),
            )
        preview = state.preview
        if preview is None:
            return {}, None, None
        return dict(preview.histogram_cards), state.repeated_bad_column, state.repeated_good_column

    def _percentile_base_record_count(self, state: ExtendMatrixTabState) -> int:
        if self._current_app_mode() == "grid_inspection":
            return len(self._grid_inspection_percentile_map_for_state(state))
        return len(self._base_records_for_state(state))

    def _update_repeated_percentile_lists(self, state: ExtendMatrixTabState) -> None:
        self._sync_correlation_limit_bounds(state)

        def summary(entries: list[tuple[FrameRecord, int, float, list[str]]]) -> tuple[int, float, int]:
            visible_entries = entries[: self._correlation_limit_for_state(state)]
            if not visible_entries:
                return 0, 0.0, 0
            frame_count = len(visible_entries)
            mean_hits = sum(count for _record, count, _avg_percentile, _labels in visible_entries) / float(frame_count)
            max_hits = max(count for _record, count, _avg_percentile, _labels in visible_entries)
            return frame_count, mean_hits, max_hits

        bad_entries = self._repeated_percentile_entries(state, band="bad")
        good_entries = self._repeated_percentile_entries(state, band="good")
        _cards, repeated_bad_column, repeated_good_column = self._percentile_widgets_for_state(state)
        if repeated_bad_column is not None and hasattr(repeated_bad_column, "set_payload"):
            frame_count, mean_hits, max_hits = summary(bad_entries)
            repeated_bad_column.set_payload(
                frame_count, mean_hits, max_hits, active=self._selected_correlation_band_for_state(state) == "bad"
            )
        if repeated_good_column is not None and hasattr(repeated_good_column, "set_payload"):
            frame_count, mean_hits, max_hits = summary(good_entries)
            repeated_good_column.set_payload(
                frame_count, mean_hits, max_hits, active=self._selected_correlation_band_for_state(state) == "good"
            )

    def _update_metric_histograms(self, state: ExtendMatrixTabState) -> None:
        histogram_cards, _bad_column, _good_column = self._percentile_widgets_for_state(state)
        if not histogram_cards:
            return
        base_record_count = self._percentile_base_record_count(state)
        available_keys = set(self._percentile_basis_keys_for_state(state, state.build_result))
        if state.percentile_filter_metric_key not in available_keys:
            state.percentile_filter_metric_key = None
            state.percentile_filter_bin_index = None
        if state.selected_percentile_metric_key not in available_keys:
            state.selected_percentile_metric_key = None
            state.selected_percentile_bin_index = None
        for metric_key, card in histogram_cards.items():
            visible = metric_key in available_keys
            if not visible:
                card.setVisible(False)
                continue
            counts = self._percentile_histogram_counts(state, metric_key)
            active_bin = self._selected_percentile_bin_for_metric(state, metric_key)
            tooltip = self._metric_hint_fallback(metric_key, state.build_result)
            card.set_payload(
                self._metric_label(metric_key, state.build_result),
                counts,
                base_record_count,
                visible=True,
                active_bin=active_bin,
                tooltip=tooltip,
            )
        self._update_repeated_percentile_lists(state)

    def _schedule_metric_histogram_update(self, state: ExtendMatrixTabState) -> None:
        self._histogram_update_generation += 1
        generation = int(self._histogram_update_generation)
        QTimer.singleShot(0, lambda s=state, g=generation: self._update_metric_histograms_chunked(s, g, 0))

    def _update_metric_histograms_chunked(self, state: ExtendMatrixTabState, generation: int, index: int) -> None:
        if generation != self._histogram_update_generation or state.widget not in self._tab_states:
            return
        histogram_cards, _bad_column, _good_column = self._percentile_widgets_for_state(state)
        if not histogram_cards:
            return
        base_record_count = self._percentile_base_record_count(state)
        available_keys = set(self._percentile_basis_keys_for_state(state, state.build_result))
        if state.percentile_filter_metric_key not in available_keys:
            state.percentile_filter_metric_key = None
            state.percentile_filter_bin_index = None
        if state.selected_percentile_metric_key not in available_keys:
            state.selected_percentile_metric_key = None
            state.selected_percentile_bin_index = None
        histogram_items = list(histogram_cards.items())
        if index >= len(histogram_items):
            QTimer.singleShot(
                0,
                lambda s=state, g=generation: (
                    self._update_repeated_percentile_lists(s)
                    if g == self._histogram_update_generation and s.widget in self._tab_states
                    else None
                ),
            )
            return
        metric_key, card = histogram_items[index]
        if metric_key not in available_keys:
            card.setVisible(False)
        else:
            counts = self._percentile_histogram_counts(state, metric_key)
            active_bin = self._selected_percentile_bin_for_metric(state, metric_key)
            tooltip = self._metric_hint_fallback(metric_key, state.build_result)
            card.set_payload(
                self._metric_label(metric_key, state.build_result),
                counts,
                base_record_count,
                visible=True,
                active_bin=active_bin,
                tooltip=tooltip,
            )
        QTimer.singleShot(0, lambda s=state, g=generation, i=index + 1: self._update_metric_histograms_chunked(s, g, i))

    def _connect_histogram_cards(self, state: ExtendMatrixTabState) -> None:
        preview = state.preview
        if preview is None:
            return
        for metric_key, card in preview.histogram_cards.items():
            if hasattr(card, "binClicked"):
                card.binClicked.connect(
                    lambda clicked_metric_key, bin_index, s=state: self._on_histogram_bin_selected(
                        s, str(clicked_metric_key), int(bin_index)
                    )
                )
            if hasattr(card, "binDoubleClicked"):
                card.binDoubleClicked.connect(
                    lambda clicked_metric_key, bin_index, s=state: self._on_histogram_bin_clicked(
                        s, str(clicked_metric_key), int(bin_index)
                    )
                )
            if hasattr(card, "binContextMenuRequested"):
                card.binContextMenuRequested.connect(
                    lambda clicked_metric_key, bin_index, global_pos, s=state: self._on_histogram_bin_context_menu(
                        s,
                        str(clicked_metric_key),
                        int(bin_index),
                        global_pos,
                    )
                )
        if state.percentile_full_matrix_check is not None:
            state.percentile_full_matrix_check.toggled.connect(
                lambda checked, s=state: self._on_percentile_display_mode_changed(s, bool(checked))
            )
        if state.correlation_limit_spin is not None:
            state.correlation_limit_spin.valueChanged.connect(
                lambda value, s=state: self._on_correlation_limit_changed(s, int(value))
            )
        if state.repeated_bad_column is not None:
            state.repeated_bad_column.columnClicked.connect(
                lambda band, s=state: self._on_correlation_column_selected(s, str(band))
            )
            if hasattr(state.repeated_bad_column, "columnDoubleClicked"):
                state.repeated_bad_column.columnDoubleClicked.connect(
                    lambda band, s=state: self._on_correlation_column_clicked(s, str(band))
                )
        if state.repeated_good_column is not None:
            state.repeated_good_column.columnClicked.connect(
                lambda band, s=state: self._on_correlation_column_selected(s, str(band))
            )
            if hasattr(state.repeated_good_column, "columnDoubleClicked"):
                state.repeated_good_column.columnDoubleClicked.connect(
                    lambda band, s=state: self._on_correlation_column_clicked(s, str(band))
                )

    def _on_correlation_column_selected(self, state: ExtendMatrixTabState, band: str) -> None:
        if str(band) not in {"bad", "good"}:
            return
        state.selected_percentile_metric_key = None
        state.selected_percentile_bin_index = None
        state.selected_correlation_band = str(band)
        self._update_repeated_percentile_lists(state)

    def _on_correlation_column_clicked(self, state: ExtendMatrixTabState, band: str) -> None:
        if str(band) not in {"bad", "good"}:
            return
        state.selected_percentile_metric_key = None
        state.selected_percentile_bin_index = None
        state.selected_correlation_band = str(band)
        state.percentile_filter_metric_key = None
        state.percentile_filter_bin_index = None
        state.correlation_filter_band = None if state.correlation_filter_band == band else str(band)
        self._apply_tab_visual_settings(state, reset_view=True)
        if state.content_tabs is not None:
            state.content_tabs.setCurrentIndex(0)

    def _on_correlation_limit_changed(self, state: ExtendMatrixTabState, value: int) -> None:
        state.correlation_limit = max(1, int(value))
        if self._correlation_filter_active(state):
            if self._current_app_mode() == "grid_inspection":
                self._refresh_grid_inspection_mode_view()
            else:
                self._apply_tab_visual_settings(state, reset_view=True)
        else:
            self._update_repeated_percentile_lists(state)

    def _on_histogram_bin_selected(self, state: ExtendMatrixTabState, metric_key: str, bin_index: int) -> None:
        state.selected_percentile_metric_key = str(metric_key)
        state.selected_percentile_bin_index = int(bin_index)
        state.selected_correlation_band = None
        histogram_cards, _bad_column, _good_column = self._percentile_widgets_for_state(state)
        base_record_count = self._percentile_base_record_count(state)
        available_keys = set(self._percentile_basis_keys_for_state(state, state.build_result))
        for card_metric_key, card in histogram_cards.items():
            if card_metric_key not in available_keys:
                card.setVisible(False)
                continue
            counts = self._percentile_histogram_counts(state, card_metric_key)
            active_bin = self._selected_percentile_bin_for_metric(state, card_metric_key)
            tooltip = self._metric_hint_fallback(card_metric_key, state.build_result)
            card.set_payload(
                self._metric_label(card_metric_key, state.build_result),
                counts,
                base_record_count,
                visible=True,
                active_bin=active_bin,
                tooltip=tooltip,
            )
        self._update_repeated_percentile_lists(state)

    def _on_histogram_bin_clicked(self, state: ExtendMatrixTabState, metric_key: str, bin_index: int) -> None:
        state.selected_percentile_metric_key = str(metric_key)
        state.selected_percentile_bin_index = int(bin_index)
        state.selected_correlation_band = None
        same_filter = state.percentile_filter_metric_key == metric_key and state.percentile_filter_bin_index == int(
            bin_index
        )
        state.correlation_filter_band = None
        if same_filter:
            state.percentile_filter_metric_key = None
            state.percentile_filter_bin_index = None
        else:
            state.percentile_filter_metric_key = str(metric_key)
            state.percentile_filter_bin_index = int(bin_index)
        self._apply_tab_visual_settings(state, reset_view=True)
        if state.content_tabs is not None:
            state.content_tabs.setCurrentIndex(0)

    def _on_histogram_bin_context_menu(
        self, state: ExtendMatrixTabState, metric_key: str, bin_index: int, global_pos
    ) -> None:
        records = self._records_for_percentile_bin(state, metric_key, bin_index)
        menu = QMenu(self._view)
        export_action = menu.addAction(self._t("context.export_percentile_frame_assets", count=len(records)))
        export_action.setEnabled(bool(records))
        selected_action = menu.exec(global_pos)
        if selected_action is export_action and records:
            self._export_percentile_records_assets(state, records, metric_key, bin_index)

    def _on_percentile_display_mode_changed(self, state: ExtendMatrixTabState, checked: bool) -> None:
        state.percentile_filter_full_matrix = bool(checked)
        self._apply_tab_visual_settings(state, reset_view=True)
        if state.content_tabs is not None:
            state.content_tabs.setCurrentIndex(0)

    def _on_grid_inspection_histogram_bin_clicked(self, metric_key: str, bin_index: int) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        state.selected_percentile_metric_key = str(metric_key)
        state.selected_percentile_bin_index = int(bin_index)
        state.selected_correlation_band = None
        same_filter = state.percentile_filter_metric_key == metric_key and state.percentile_filter_bin_index == int(
            bin_index
        )
        state.correlation_filter_band = None
        if same_filter:
            state.percentile_filter_metric_key = None
            state.percentile_filter_bin_index = None
        else:
            state.percentile_filter_metric_key = str(metric_key)
            state.percentile_filter_bin_index = int(bin_index)
        self._refresh_grid_inspection_mode_view()
        if hasattr(self, "grid_inspection_content_tabs"):
            self.grid_inspection_content_tabs.setCurrentIndex(0)

    def _on_grid_inspection_histogram_bin_selected(self, metric_key: str, bin_index: int) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        self._on_histogram_bin_selected(state, metric_key, bin_index)

    def _on_grid_inspection_histogram_bin_context_menu(self, metric_key: str, bin_index: int, global_pos) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        self._on_histogram_bin_context_menu(state, metric_key, bin_index, global_pos)

    def _on_grid_inspection_percentile_display_mode_changed(self, checked: bool) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        state.percentile_filter_full_matrix = bool(checked)
        self._refresh_grid_inspection_mode_view()
        if hasattr(self, "grid_inspection_content_tabs"):
            self.grid_inspection_content_tabs.setCurrentIndex(0)

    def _on_grid_inspection_correlation_column_clicked(self, band: str) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        if str(band) not in {"bad", "good"}:
            return
        state.selected_percentile_metric_key = None
        state.selected_percentile_bin_index = None
        state.selected_correlation_band = str(band)
        state.percentile_filter_metric_key = None
        state.percentile_filter_bin_index = None
        state.correlation_filter_band = None if state.correlation_filter_band == band else str(band)
        self._refresh_grid_inspection_mode_view()
        if hasattr(self, "grid_inspection_content_tabs"):
            self.grid_inspection_content_tabs.setCurrentIndex(0)

    def _on_grid_inspection_correlation_column_selected(self, band: str) -> None:
        state = self._current_tab_state()
        if state is None or str(band) not in {"bad", "good"}:
            return
        state.selected_percentile_metric_key = None
        state.selected_percentile_bin_index = None
        state.selected_correlation_band = str(band)
        self._update_repeated_percentile_lists(state)

    def _on_grid_inspection_correlation_limit_changed(self, value: int) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        state.correlation_limit_spin = getattr(self, "grid_inspection_correlation_limit_spin", None)
        self._on_correlation_limit_changed(state, int(value))

    def _model_display_name(self, state: ExtendMatrixTabState, model_id: str) -> str:
        for spec in state.build_result.model_specs:
            if spec.model_id == model_id:
                return spec.display_name
        return model_id

    def _format_component_value(self, value) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _metric_component_lines(self, state: ExtendMatrixTabState, record: FrameRecord, metric_key: str) -> list[str]:
        summary = record.summary
        if summary is None:
            return []
        parsed_pair = parse_pair_metric_key(metric_key)
        if parsed_pair is not None:
            model_a, model_b, operation = parsed_pair
            left = self._model_display_name(state, model_a)
            right = self._model_display_name(state, model_b)
            pair_row = None
            for row in summary.pairwise_metrics:
                row_a = str(row.get("model_a", ""))
                row_b = str(row.get("model_b", ""))
                if {row_a, row_b} == {model_a, model_b}:
                    pair_row = row
                    break
            lines = [
                f"pair: {left} -> {right}",
                f"operation: {PAIR_OPERATION_LABELS.get(operation, operation)}",
                f"value: {self._format_component_value(summary.metric_values.get(metric_key))}",
            ]
            if pair_row is not None:
                for key in ("iou", "dice", "agreement_score", "soft_iou", "soft_dice"):
                    if key in pair_row:
                        lines.append(f"{key}: {self._format_component_value(pair_row.get(key))}")
            return lines
        if metric_key in {"overall_frame_score", "export_priority_score"}:
            return [
                "source: model comparison",
                "formula: disagreement_score",
                f"disagreement_score: {summary.disagreement_score:.4f}",
            ]
        if metric_key == "model_model_score":
            lines = []
            for row in summary.pairwise_metrics[:8]:
                left = self._model_display_name(state, str(row.get("model_a", "-")))
                right = self._model_display_name(state, str(row.get("model_b", "-")))
                agreement = float(row.get("agreement_score", 0.0))
                lines.append(f"{left} vs {right}: {agreement:.4f}")
                if summary.frame_type == "polygon":
                    lines.append(f"  active soft_dice: {float(row.get('soft_dice', 0.0)):.4f}")
                    lines.append(f"  active soft_iou: {float(row.get('soft_iou', 0.0)):.4f}")
                    lines.append(f"  active ssim: {float(row.get('ssim', 0.0)):.4f}")
                    lines.append(f"  active dice: {float(row.get('dice', 0.0)):.4f}")
                    lines.append(f"  active iou: {float(row.get('iou', 0.0)):.4f}")
                    lines.append(f"  active hausdorff_distance: {float(row.get('hausdorff_distance', 0.0)):.4f}")
                    lines.append(f"  active centroid_distance: {float(row.get('centroid_distance', 0.0)):.4f}")
                    lines.append(f"  auxiliary mae: {float(row.get('mae', 0.0)):.4f}")
                    lines.append(f"  auxiliary rmse: {float(row.get('rmse', 0.0)):.4f}")
                    lines.append(f"  auxiliary count_agreement: {float(row.get('count_agreement', 0.0)):.4f}")
                else:
                    lines.append(f"  active precision: {float(row.get('precision', 0.0)):.4f}")
                    lines.append(f"  active recall: {float(row.get('recall', 0.0)):.4f}")
                    lines.append(f"  active f1_at_r: {float(row.get('f1', 0.0)):.4f}")
                    lines.append(
                        f"  active mean_localization_error: {float(row.get('mean_localization_error', 0.0)):.4f}"
                    )
                    lines.append(
                        f"  active localization_agreement: {float(row.get('localization_agreement', 0.0)):.4f}"
                    )
                    lines.append(f"  active count_agreement: {float(row.get('count_agreement', 0.0)):.4f}")
            return lines or [f"disagreement_score: {summary.disagreement_score:.4f}"]
        if metric_key == "disagreement_score":
            return [
                "formula: 1 - model_model_score",
                f"model_model_score: {summary.metric_values.get('model_model_score', 0.0):.4f}",
            ]
        if metric_key in {
            "overall_polygon_score",
            "iou_score",
            "dice_score",
            "polygon_bce_score",
            "iou",
            "dice",
            "bce",
        }:
            return [
                f"iou: {self._format_component_value(summary.metric_values.get('iou'))}",
                f"dice: {self._format_component_value(summary.metric_values.get('dice'))}",
                f"bce: {self._format_component_value(summary.metric_values.get('bce'))}",
                f"iou_score: {self._format_component_value(summary.metric_values.get('iou_score'))}",
                f"dice_score: {self._format_component_value(summary.metric_values.get('dice_score'))}",
                f"polygon_bce_score: {self._format_component_value(summary.metric_values.get('polygon_bce_score'))}",
                f"overall_polygon_score: {self._format_component_value(summary.metric_values.get('overall_polygon_score'))}",
            ]
        if metric_key in {
            "overall_point_score",
            "precision_score",
            "recall_score",
            "f1_score",
            "localization_score",
            "precision",
            "recall",
            "f1",
            "mean_localization_distance",
        }:
            return [
                f"precision: {self._format_component_value(summary.metric_values.get('precision'))}",
                f"recall: {self._format_component_value(summary.metric_values.get('recall'))}",
                f"f1: {self._format_component_value(summary.metric_values.get('f1'))}",
                f"mean_localization_distance: {self._format_component_value(summary.metric_values.get('mean_localization_distance'))}",
                f"tp: {self._format_component_value(summary.metric_values.get('tp'))}",
                f"fp: {self._format_component_value(summary.metric_values.get('fp'))}",
                f"fn: {self._format_component_value(summary.metric_values.get('fn'))}",
                f"precision_score: {self._format_component_value(summary.metric_values.get('precision_score'))}",
                f"recall_score: {self._format_component_value(summary.metric_values.get('recall_score'))}",
                f"f1_score: {self._format_component_value(summary.metric_values.get('f1_score'))}",
                f"localization_score: {self._format_component_value(summary.metric_values.get('localization_score'))}",
                f"overall_point_score: {self._format_component_value(summary.metric_values.get('overall_point_score'))}",
            ]
        parsed_metric = metric_key.split("::", 1) if "::" in str(metric_key) else None
        if parsed_metric is not None:
            family, model_id = parsed_metric
            confidence_row = summary.model_confidence.get(model_id) if summary.model_confidence is not None else None
            confidence_output_row = (
                getattr(summary, "model_confidence_output", {}).get(model_id)
                if getattr(summary, "model_confidence_output", None) is not None
                else None
            )
            model_name = self._model_display_name(state, model_id)
            if family == "model_confidence" and confidence_row is not None:
                if hasattr(confidence_row, "mean_object_confidence"):
                    return [
                        f"model: {model_name}",
                        f"frame_uncertainty_score: {self._format_component_value(getattr(confidence_row, 'frame_uncertainty_score', None))}",
                        f"summary_metric: {self._format_component_value(getattr(confidence_row, 'summary_metric', None))}",
                        f"mean_uncertainty: {self._format_component_value(getattr(confidence_row, 'mean_uncertainty', None))}",
                        f"low_conf_fraction: {self._format_component_value(getattr(confidence_row, 'low_conf_fraction', None))}",
                        f"worst_tail_uncertainty: {self._format_component_value(getattr(confidence_row, 'worst_tail_uncertainty', None))}",
                        f"largest_low_conf_component: {self._format_component_value(getattr(confidence_row, 'largest_low_conf_component', None))}",
                        f"mean_object_confidence: {self._format_component_value(getattr(confidence_row, 'mean_object_confidence', None))}",
                        f"uncertain_support_fraction: {self._format_component_value(getattr(confidence_row, 'uncertain_support_fraction', None))}",
                        f"top_uncertainty_mean: {self._format_component_value(getattr(confidence_row, 'top_uncertainty_mean', None))}",
                        f"largest_uncertain_region_fraction: {self._format_component_value(getattr(confidence_row, 'largest_uncertain_region_fraction', None))}",
                        f"mean_core_confidence: {self._format_component_value(getattr(confidence_row, 'mean_core_confidence', None))}",
                        f"mean_boundary_uncertainty: {self._format_component_value(getattr(confidence_row, 'mean_boundary_uncertainty', None))}",
                        f"mean_weighted_confidence: {self._format_component_value(getattr(confidence_row, 'mean_weighted_confidence', None))}",
                        f"mean_object_probability: {self._format_component_value(getattr(confidence_row, 'mean_object_probability', None))}",
                        f"uncertain_fraction: {self._format_component_value(getattr(confidence_row, 'uncertain_fraction', None))}",
                        f"mean_transition_width: {self._format_component_value(getattr(confidence_row, 'mean_transition_width', None))}",
                        f"polygon_count: {self._format_component_value(getattr(confidence_row, 'polygon_count', None))}",
                    ]
                return [
                    f"model: {model_name}",
                    f"frame_uncertainty_score: {self._format_component_value(getattr(confidence_row, 'frame_uncertainty_score', None))}",
                    f"mean_uncertainty: {self._format_component_value(getattr(confidence_row, 'mean_uncertainty', None))}",
                    f"low_conf_fraction: {self._format_component_value(getattr(confidence_row, 'low_conf_fraction', None))}",
                    f"worst_tail_uncertainty: {self._format_component_value(getattr(confidence_row, 'worst_tail_uncertainty', None))}",
                    f"largest_low_conf_component: {self._format_component_value(getattr(confidence_row, 'largest_low_conf_component', None))}",
                    f"mean_point_confidence: {self._format_component_value(getattr(confidence_row, 'mean_point_confidence', None))}",
                    f"uncertain_support_fraction: {self._format_component_value(getattr(confidence_row, 'uncertain_support_fraction', None))}",
                    f"top_uncertainty_mean: {self._format_component_value(getattr(confidence_row, 'top_uncertainty_mean', None))}",
                    f"largest_uncertain_region_fraction: {self._format_component_value(getattr(confidence_row, 'largest_uncertain_region_fraction', None))}",
                    f"mean_center_confidence: {self._format_component_value(getattr(confidence_row, 'mean_center_confidence', None))}",
                    f"mean_local_confidence: {self._format_component_value(getattr(confidence_row, 'mean_local_confidence', None))}",
                    f"mean_point_probability: {self._format_component_value(getattr(confidence_row, 'mean_point_probability', None))}",
                    f"mean_point_contrast: {self._format_component_value(getattr(confidence_row, 'mean_point_contrast', None))}",
                    f"point_count: {self._format_component_value(getattr(confidence_row, 'point_count', None))}",
                ]
            if family == "model_output_confidence" and confidence_output_row is not None:
                return [
                    f"model: {model_name}",
                    f"frame_uncertainty_score: {self._format_component_value(getattr(confidence_output_row, 'frame_uncertainty_score', None))}",
                    f"mean_confidence: {self._format_component_value(getattr(confidence_output_row, 'mean_confidence', None))}",
                    f"mean_uncertainty: {self._format_component_value(getattr(confidence_output_row, 'mean_uncertainty', None))}",
                    f"uncertain_fraction: {self._format_component_value(getattr(confidence_output_row, 'uncertain_fraction', None))}",
                    f"top_uncertainty_mean: {self._format_component_value(getattr(confidence_output_row, 'top_uncertainty_mean', None))}",
                    f"largest_uncertain_region_fraction: {self._format_component_value(getattr(confidence_output_row, 'largest_uncertain_region_fraction', None))}",
                    f"min_confidence: {self._format_component_value(getattr(confidence_output_row, 'min_confidence', None))}",
                    f"max_confidence: {self._format_component_value(getattr(confidence_output_row, 'max_confidence', None))}",
                ]
            if family == "model_uncertain_fraction" and confidence_row is not None:
                return [
                    f"model: {model_name}",
                    f"uncertain_fraction: {self._format_component_value(getattr(confidence_row, 'uncertain_fraction', None))}",
                    f"mean_boundary_uncertainty: {self._format_component_value(getattr(confidence_row, 'mean_boundary_uncertainty', None))}",
                    f"mean_transition_width: {self._format_component_value(getattr(confidence_row, 'mean_transition_width', None))}",
                    f"mean_core_confidence: {self._format_component_value(getattr(confidence_row, 'mean_core_confidence', None))}",
                ]
            if family == "model_point_contrast" and confidence_row is not None:
                return [
                    f"model: {model_name}",
                    f"mean_point_contrast: {self._format_component_value(getattr(confidence_row, 'mean_point_contrast', None))}",
                    f"mean_local_confidence: {self._format_component_value(getattr(confidence_row, 'mean_local_confidence', None))}",
                    f"mean_center_confidence: {self._format_component_value(getattr(confidence_row, 'mean_center_confidence', None))}",
                    f"point_count: {self._format_component_value(getattr(confidence_row, 'point_count', None))}",
                ]
        return []

    def _overall_score_style(self, value: float | None) -> str:
        if value is None:
            background = "#2f3844"
            foreground = "#edf3fb"
        elif value < 0.33:
            background = "#1f5f3b"
            foreground = "#e9fff1"
        elif value < 0.66:
            background = "#8a6a12"
            foreground = "#fff7da"
        else:
            background = "#8c2f39"
            foreground = "#ffe9ec"
        return f"padding: 6px 10px; border-radius: 8px; background-color: {background}; color: {foreground}; font-weight: 700;"

    def _overall_score_text(self, value: float | None) -> str:
        if value is None:
            return "-"
        if value < 0.33:
            level = "LOW"
        elif value < 0.66:
            level = "MEDIUM"
        else:
            level = "HIGH"
        return f"{level} {value:.4f}"

    def _show_progress_bar(
        self, *, visible: bool, current: int = 0, total: int = 0, key: str = "", format_text: str | None = None
    ) -> None:
        if not visible:
            self.build_progress.hide()
            self.build_progress.setRange(0, 1)
            self.build_progress.setValue(0)
            return
        if total > 0:
            self.build_progress.setRange(0, total)
            self.build_progress.setValue(min(current, total))
            self.build_progress.setFormat(format_text or f"{current}/{total}")
        else:
            self.build_progress.setRange(0, 0)
            self.build_progress.setFormat(format_text or "Working...")
        self.build_progress.setToolTip(key)
        self.build_progress.show()

    def _compact_folder_label(self, folder: FolderSpec | None) -> tuple[str, str]:
        if folder is None:
            return "not set", ""
        path = folder.path
        tail = path.name
        parent = path.parent.name if path.parent != path else ""
        short = f"{parent}/{tail}" if parent else tail
        return short, str(path)

    @staticmethod
    def _compact_path_text(path_text: str | None) -> str:
        if not path_text:
            return ""
        path = Path(str(path_text))
        tail = path.name or str(path)
        parent = path.parent.name if path.parent != path else ""
        return f"{parent}/{tail}" if parent else tail

    def _update_source_labels(self) -> None:
        original_text, original_tooltip = self._compact_folder_label(self._original_folder)
        export_text = (
            self._compact_path_text(str(self._export_folder)) if self._export_folder is not None else "not set"
        )
        export_tooltip = "" if self._export_folder is None else str(self._export_folder)
        self.original_folder_value.setText(original_text)
        self.original_folder_value.setToolTip(original_tooltip)
        self.export_folder_value.setText(export_text)
        self.export_folder_value.setToolTip(export_tooltip)

    def _add_folder(self) -> None:
        if self._worker_thread is not None:
            return
        folder = QFileDialog.getExistingDirectory(self._view, self._t("dialog.select_model_folder"))
        if not folder:
            return
        folder_path = Path(folder)
        if not self._folder_has_supported_images(folder_path):
            QMessageBox.warning(
                self._view,
                self._t("dialog.warning_title"),
                f"Folder has no supported images: {folder_path}",
            )
            return
        item = self._append_folder_item(folder_path, checked=True)
        self.folder_list.setCurrentItem(item)
        self._refresh_folder_rows()
        self._refresh_pair_matrix()
        self._sync_action_buttons()

    def _clear_folders(self) -> None:
        self.folder_list.clear()
        self._comparison_pair_operations.clear()
        self._pair_defaults_initialized = False
        self._refresh_folder_rows()
        self._refresh_pair_matrix()
        self._sync_action_buttons()

    def _set_original_folder(self) -> None:
        if self._worker_thread is not None:
            return
        folder = QFileDialog.getExistingDirectory(self._view, self._t("dialog.select_original_folder"))
        if not folder:
            return
        path = Path(folder)
        if not self._folder_has_supported_images(path):
            QMessageBox.warning(
                self._view,
                self._t("dialog.warning_title"),
                f"Base folder has no supported images: {path}",
            )
            return
        self._original_folder = FolderSpec(path=path, label=path.name)
        self._update_source_labels()
        self._sync_action_buttons()

    def _clear_original_folder(self) -> None:
        self._original_folder = None
        self._update_source_labels()
        self._sync_action_buttons()

    def _bootstrap_source_folder_for_sampling(self) -> Path | None:
        if self._original_folder is None:
            return None
        path = Path(self._original_folder.path)
        if not path.exists():
            return None
        return path

    def _set_export_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self._view, self._t("dialog.select_export_folder"))
        if not folder:
            return
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        self._export_folder = path
        self._update_source_labels()
        self._sync_action_buttons()

    def _clear_export_folder(self) -> None:
        self._export_folder = None
        self._update_source_labels()
        self._sync_action_buttons()

    def _current_app_mode(self) -> str:
        return str(self.app_mode_combo.currentData() or "validation")

    def _on_build_requested(self) -> None:
        state = self._current_tab_state()
        active_model_count = len(self._checked_model_specs())
        can_build_from_base_only = (
            active_model_count <= 0 and self._original_folder is not None and Path(self._original_folder.path).exists()
        )
        if state is None or active_model_count >= self._required_model_count_for_build() or can_build_from_base_only:
            self._start_build()
            return
        self._apply_pending_display_controls(state)
        ok = self._apply_tab_visual_settings(state, reset_view=True, update_histograms=False)
        if not ok:
            return
        self._sync_action_buttons()

    def _on_app_mode_changed(self) -> None:
        mode = self._current_app_mode()
        self._app_mode_refresh_generation += 1
        refresh_generation = int(self._app_mode_refresh_generation)
        if hasattr(self._view, "_set_app_mode"):
            self._view._set_app_mode(mode)
        if mode == "grid_inspection":
            QTimer.singleShot(0, lambda g=refresh_generation: self._refresh_grid_inspection_mode_view_deferred(g))
        self._sync_action_buttons()

    def _refresh_grid_inspection_mode_view_deferred(self, generation: int) -> None:
        if generation != self._app_mode_refresh_generation or self._current_app_mode() != "grid_inspection":
            return
        try:
            self._refresh_grid_inspection_mode_view()
        except Exception as error:
            QMessageBox.critical(self._view, self._t("dialog.warning_title"), str(error))

    def _refresh_grid_inspection_mode_view(self) -> None:
        if not hasattr(self, "grid_inspection_matrix_view"):
            return
        state = self._current_tab_state()
        if state is None or not getattr(state.build_result, "records", None):
            self._sync_grid_inspection_layer_tabs(None)
            for view in self._grid_inspection_views().values():
                view.set_grid_inspection_payloads({}, enabled=False)
                view.set_highlighted_record_keys(set())
                view.set_records([], sort_mode="name", reset_view=True)
            self._sync_grid_reference_controls(None)
            self._refresh_grid_inspection_errors_panel(None)
            return
        views = self._grid_inspection_views()
        self._sync_grid_inspection_layer_tabs(state)
        for view in views.values():
            view.set_layout_config(state.layout_config)
            view.set_excluded_record_keys(set(state.excluded_record_keys))
            view.set_grid_inspection_visual_mode(True)
            view.set_gradient_preset(state.gradient_name or DEFAULT_GRADIENT_NAME)
            view.set_score_view_mode(
                str(
                    state.matrix_score_view_mode
                    or self.matrix_score_view_combo.currentData()
                    or DEFAULT_MATRIX_SCORE_VIEW_MODE
                )
            )
            view.set_cell_size(int(getattr(state, "cell_size", DEFAULT_CELL_SIZE)))
        grid_full_matrix_check = getattr(self, "grid_inspection_percentile_full_matrix_check", None)
        if grid_full_matrix_check is not None:
            state.percentile_filter_full_matrix = bool(grid_full_matrix_check.isChecked())
        grid_correlation_limit_spin = getattr(self, "grid_inspection_correlation_limit_spin", None)
        if grid_correlation_limit_spin is not None:
            state.correlation_limit_spin = grid_correlation_limit_spin
            state.correlation_limit = self._correlation_limit_for_state(state)
        layer_payloads = getattr(state, "grid_inspection_payloads_by_layer", {}) or {}
        active_layer = str(getattr(state, "grid_inspection_layer", "confidence") or "confidence")
        if not layer_payloads and getattr(state, "grid_inspection_payload_by_key", None):
            layer_payloads = {active_layer: dict(state.grid_inspection_payload_by_key)}
        state.grid_inspection_payload_by_key = dict(layer_payloads.get(active_layer, {}) or {})
        records = list(self._display_records_for_state(state))
        highlight_keys = self._matrix_highlight_keys_for_state(state)
        for layer_key, view in views.items():
            payloads = (
                dict(layer_payloads.get(layer_key, {}) or {})
                if bool(getattr(state, "grid_inspection_results_ready", False))
                else {}
            )
            view.set_grid_inspection_payloads(payloads, enabled=True)
            view.set_highlighted_record_keys(highlight_keys)
            view.set_records(records, sort_mode="name", reset_view=True)
        self._sync_grid_reference_controls(state)
        QTimer.singleShot(50, lambda s=state: self._finish_grid_inspection_summary_refresh(s))

    def _grid_inspection_error_type(self, status: str, reasons: tuple[str, ...]) -> str:
        reason_set = {str(reason) for reason in reasons}
        for _label_key, reason in GRID_INSPECTION_ERROR_TYPE_OPTIONS:
            if reason in reason_set:
                return reason
        return str(status or "broken")

    def _grid_inspection_error_type_label(self, error_type: str) -> str:
        labels = {str(value): self._t(label_key) for label_key, value in GRID_INSPECTION_ERROR_TYPE_OPTIONS}
        labels.update(
            {
                "broken": self._t("grid_status.broken"),
                "suspicious": self._t("grid_status.suspicious"),
                "artifact": self._t("grid_status.artifact"),
            }
        )
        return labels.get(str(error_type), str(error_type or self._t("grid_error.defect")))

    def _grid_inspection_error_filter(self) -> str:
        combo = getattr(self, "grid_inspection_error_filter", None)
        if combo is None:
            return "all"
        return str(combo.currentData() or "all")

    def _grid_inspection_error_filter_matches(self, payload: dict[str, object], filter_key: str) -> bool:
        if str(filter_key or "all") == "all":
            return True
        error_type = str(payload.get("error_type") or "")
        reasons = {str(reason) for reason in payload.get("reasons", ()) or ()}
        return error_type == filter_key or str(filter_key) in reasons

    def _iter_grid_inspection_error_payloads(self, state: ExtendMatrixTabState | None):
        if state is None or not bool(getattr(state, "grid_inspection_results_ready", False)):
            return
        excluded_keys = self._excluded_record_keys_for_state(state)
        records_by_key = {str(record.key): record for record in state.build_result.records}
        visible_keys: set[str] | None = None
        if (
            state.percentile_filter_metric_key is not None and state.percentile_filter_bin_index is not None
        ) or state.correlation_filter_band in {"bad", "good"}:
            visible_keys = {str(record.key) for record in self._display_records_for_state(state)}
        for record_key, result in sorted(
            (getattr(state, "grid_inspection_payload_by_key", {}) or {}).items(), key=lambda item: str(item[0])
        ):
            if str(record_key) in excluded_keys:
                continue
            if visible_keys is not None and str(record_key) not in visible_keys:
                continue
            record = records_by_key.get(str(record_key))
            if record is None:
                continue
            for index, cell in enumerate(getattr(result, "per_cell_results", getattr(result, "cells", ())) or ()):
                status = str(getattr(cell, "status", "") or "")
                if status == "normal":
                    continue
                reasons = tuple(str(reason) for reason in (getattr(cell, "reasons", ()) or ()))
                error_type = self._grid_inspection_error_type(status, reasons)
                bbox = tuple(int(value) for value in (getattr(cell, "bbox", (0, 0, 1, 1)) or (0, 0, 1, 1))[:4])
                feature_cluster_id = getattr(cell, "feature_cluster_id", None)
                feature_cluster_label = str(getattr(cell, "feature_cluster_label", "") or "")
                yield {
                    "record_key": str(record_key),
                    "record_name": str(getattr(record, "display_name", "") or record_key),
                    "cell_index": int(index),
                    "row": int(getattr(cell, "row", -1)),
                    "col": int(getattr(cell, "col", getattr(cell, "column", -1))),
                    "bbox": bbox,
                    "score": float(getattr(cell, "score", 0.0)),
                    "status": status,
                    "reasons": reasons,
                    "error_type": error_type,
                    "feature_cluster_id": None if feature_cluster_id is None else int(feature_cluster_id),
                    "feature_cluster_label": feature_cluster_label,
                }

    def _grid_inspection_error_payloads(self, state: ExtendMatrixTabState | None) -> list[dict[str, object]]:
        items = list(self._iter_grid_inspection_error_payloads(state))
        return sorted(
            items,
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("record_name", "")),
                int(item.get("cell_index", 0)),
            ),
        )

    def _refresh_grid_inspection_errors_panel(self, state: ExtendMatrixTabState | None = None) -> None:
        error_list = getattr(self, "grid_inspection_error_list", None)
        counter = getattr(self, "grid_inspection_error_counter", None)
        if error_list is None or counter is None:
            return
        if state is None:
            state = self._current_tab_state()
        filter_key = self._grid_inspection_error_filter()
        total_count = 0
        ranked: list[tuple[float, int, dict[str, object]]] = []
        for index, payload in enumerate(self._iter_grid_inspection_error_payloads(state)):
            total_count += 1
            if not self._grid_inspection_error_filter_matches(payload, filter_key):
                continue
            entry = (float(payload.get("score", 0.0)), -index, payload)
            if len(ranked) < GRID_INSPECTION_ERROR_LIST_LIMIT:
                heappush(ranked, entry)
            elif entry[:2] > ranked[0][:2]:
                heapreplace(ranked, entry)
        visible_items = [entry[2] for entry in sorted(ranked, key=lambda entry: (-entry[0], -entry[1]))]
        error_list.blockSignals(True)
        try:
            error_list.clear()
            for payload in visible_items:
                bbox = tuple(payload.get("bbox", (0, 0, 1, 1)) or (0, 0, 1, 1))
                label = self._grid_inspection_error_type_label(str(payload.get("error_type") or ""))
                cluster_label = str(payload.get("feature_cluster_label") or "")
                cluster_text = f" | {cluster_label}" if cluster_label else ""
                text = (
                    f"{payload.get('record_name', '-')}\n"
                    f"{label} | score {float(payload.get('score', 0.0)):.2f} | "
                    f"r{int(payload.get('row', -1))} c{int(payload.get('col', -1))} | "
                    f"bbox {bbox}{cluster_text}"
                )
                item = QListWidgetItem(text)
                item.setIcon(grid_inspection_error_type_icon(str(payload.get("error_type") or "")))
                item.setData(Qt.ItemDataRole.UserRole, dict(payload))
                reason_text = ", ".join(str(reason) for reason in payload.get("reasons", ()) or ())
                tooltip_parts = [part for part in (reason_text, cluster_label) if part]
                item.setToolTip(" | ".join(tooltip_parts) or label)
                error_list.addItem(item)
        finally:
            error_list.blockSignals(False)
        counter.setText(self._t("grid_errors.counter", visible=len(visible_items), total=total_count))

    def _on_grid_inspection_error_filter_changed(self, *_args) -> None:
        self._refresh_grid_inspection_errors_panel(self._current_tab_state())

    def _on_grid_inspection_error_item_clicked(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        payload = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            return
        state = self._current_tab_state()
        if state is None:
            return
        record_key = str(payload.get("record_key") or "")
        record = next((entry for entry in state.build_result.records if str(entry.key) == record_key), None)
        if record is None:
            return
        self.grid_inspection_matrix_view.select_record_by_key(record.key, ensure_visible=True)
        self._open_record_details(record, state, grid_focus=dict(payload))

    def _on_grid_inspection_record_selected(self, record: FrameRecord | None) -> None:
        state = self._current_tab_state()
        if state is not None:
            self._update_matrix_preview(state, record)

    def _on_grid_inspection_record_activated(self, record: FrameRecord | None) -> None:
        if record is None:
            return
        state = self._current_tab_state()
        if state is None:
            return
        self._open_record_details(record, state)

    def _grid_inspection_selected_records(self, fallback_record: FrameRecord | None = None) -> tuple[FrameRecord, ...]:
        if not hasattr(self, "grid_inspection_matrix_view"):
            return (fallback_record,) if fallback_record is not None else ()
        selected: tuple[FrameRecord, ...] = tuple()
        try:
            selected = tuple(self.grid_inspection_matrix_view.selected_records())
        except Exception:
            selected = tuple()
        if selected:
            return selected
        if fallback_record is not None:
            return (fallback_record,)
        return tuple()

    def _grid_inspection_export_selected_records(self, state: ExtendMatrixTabState) -> tuple[FrameRecord, ...]:
        selected_by_key: dict[str, FrameRecord] = {}
        state_selected_records: tuple[FrameRecord, ...] = tuple()
        try:
            state_selected_records = tuple(state.matrix_view.selected_records()) if state is not None else tuple()
        except Exception:
            state_selected_records = tuple()
        for source_records in (self._grid_inspection_selected_records(), state_selected_records):
            for record in source_records:
                key = str(getattr(record, "key", "") or "")
                if key and key not in selected_by_key:
                    selected_by_key[key] = record
        return tuple(selected_by_key.values())

    def _grid_inspection_record_by_key(self, state: ExtendMatrixTabState | None, key: str | None) -> FrameRecord | None:
        if state is None or not key:
            return None
        target_key = str(key)
        for record in getattr(state.build_result, "records", ()) or ():
            if str(getattr(record, "key", "") or "") == target_key:
                return record
        return None

    def _grid_inspection_reference_record(self, state: ExtendMatrixTabState | None) -> FrameRecord | None:
        return self._grid_inspection_record_by_key(
            state,
            getattr(state, "grid_inspection_reference_record_key", None) if state is not None else None,
        )

    def _grid_inspection_reference_display_name(self, record: FrameRecord | None) -> str:
        if record is None:
            return ""
        return str(getattr(record, "display_name", "") or getattr(record, "key", "") or "")

    def _sync_grid_reference_controls(self, state: ExtendMatrixTabState | None = None) -> None:
        record = self._grid_inspection_reference_record(state)
        key = str(getattr(record, "key", "") or "")
        stale_key = str(getattr(state, "grid_inspection_reference_record_key", "") or "") if state is not None else ""
        label = getattr(self, "grid_reference_frame_label", None)
        if label is not None:
            if record is not None:
                label.setText(
                    self._t("grid_reference.selected", name=self._grid_inspection_reference_display_name(record))
                )
            elif stale_key:
                label.setText(self._t("grid_reference.missing", key=stale_key))
            else:
                label.setText(self._t("grid_reference.none"))
        clear_button = getattr(self, "grid_reference_frame_clear_button", None)
        if clear_button is not None:
            clear_button.setEnabled(bool(stale_key))
        for view in self._grid_inspection_views().values():
            if hasattr(view, "set_reference_key"):
                view.set_reference_key(key or None)

    def _on_grid_reference_select_requested(self) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        record = None
        for view in (getattr(self, "grid_inspection_matrix_view", None), getattr(state, "matrix_view", None)):
            if view is None or not hasattr(view, "current_record"):
                continue
            try:
                record = view.current_record()
            except Exception:
                record = None
            if record is not None:
                break
        if record is None:
            selected_records = self._grid_inspection_export_selected_records(state)
            record = selected_records[0] if len(selected_records) == 1 else None
        if record is None:
            QMessageBox.information(
                self._view,
                self._t("dialog.info_title"),
                self._t("grid_reference.select_one"),
            )
            return
        state.grid_inspection_reference_record_key = str(getattr(record, "key", "") or "") or None
        self._sync_grid_reference_controls(state)

    def _on_grid_reference_clear_requested(self) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        state.grid_inspection_reference_record_key = None
        self._sync_grid_reference_controls(state)

    def _grid_inspection_reference_profile_for_state(
        self,
        state: ExtendMatrixTabState,
        config: GridDamageAnalysisConfig,
    ) -> GridCellReferenceProfile | None:
        record = self._grid_inspection_reference_record(state)
        if record is None:
            stale_key = str(getattr(state, "grid_inspection_reference_record_key", "") or "")
            if stale_key:
                QMessageBox.warning(
                    self._view,
                    self._t("dialog.warning_title"),
                    self._t("grid_reference.record_missing", key=stale_key),
                )
            return None
        model_id = self._grid_inspection_model_id_for_state(state)
        path_text = self._grid_inspection_layer_source_path_for_record(record, model_id, "confidence")
        if not path_text or not Path(path_text).is_file():
            QMessageBox.warning(
                self._view,
                self._t("dialog.warning_title"),
                self._t("grid_reference.file_missing", name=self._grid_inspection_reference_display_name(record)),
            )
            return None
        profile = build_grid_cell_reference_profile_path(path_text, frame_id=str(record.key), config=config)
        if profile is None:
            QMessageBox.warning(
                self._view,
                self._t("dialog.warning_title"),
                self._t("grid_reference.profile_failed", name=self._grid_inspection_reference_display_name(record)),
            )
            return None
        return profile

    def _current_matrix_selected_records(self) -> tuple[FrameRecord, ...]:
        if self._current_app_mode() == "grid_inspection":
            return self._grid_inspection_selected_records()
        state = self._current_tab_state()
        if state is None:
            return tuple()
        try:
            return tuple(state.matrix_view.selected_records())
        except Exception:
            return tuple()

    def _grid_inspection_calculation_scope_records(self, state: ExtendMatrixTabState) -> tuple[FrameRecord, ...]:
        scope_keys = {str(key) for key in getattr(state, "grid_inspection_calculation_record_keys", set()) if str(key)}
        if not scope_keys:
            return tuple()
        return tuple(record for record in getattr(state.build_result, "records", ()) if str(record.key) in scope_keys)

    @staticmethod
    def _frame_search_number_groups(record: FrameRecord) -> tuple[int, ...]:
        values: list[int] = []
        identity = getattr(record, "identity", None)
        for value in (
            getattr(identity, "frame_id", None) if identity is not None else None,
            getattr(identity, "base_id", None) if identity is not None else None,
        ):
            try:
                if value is not None:
                    values.append(int(value))
            except (TypeError, ValueError) as error:
                _LOGGER.debug("Ignoring invalid frame identity value %r: %s", value, error)
        for text in (getattr(record, "key", ""), getattr(record, "display_name", "")):
            for group in re.findall(r"\d+", str(text or "")):
                try:
                    values.append(int(group))
                except (TypeError, ValueError):
                    continue
        return tuple(values)

    @staticmethod
    def _frame_search_match_rank(record: FrameRecord, query: str, query_number: int | None) -> int | None:
        query_text = str(query or "").strip().lower()
        if not query_text:
            return None
        identity = getattr(record, "identity", None)
        if query_number is not None:
            for value in (
                getattr(identity, "frame_id", None) if identity is not None else None,
                getattr(identity, "base_id", None) if identity is not None else None,
            ):
                try:
                    if value is not None and int(value) == int(query_number):
                        return 0
                except (TypeError, ValueError):
                    continue
            if int(query_number) in KarakalPresenter._frame_search_number_groups(record):
                return 1
        texts = (
            str(getattr(record, "key", "") or "").lower(),
            str(getattr(record, "display_name", "") or "").lower(),
        )
        if query_text in texts:
            return 2
        if any(query_text in text for text in texts):
            return 3
        return None

    def _frame_search_view_and_records(self, state: ExtendMatrixTabState):
        mode = self._current_app_mode()
        if mode == "grid_inspection" and hasattr(self, "grid_inspection_matrix_view"):
            return self.grid_inspection_matrix_view, tuple(self._display_records_for_state(state))
        return state.matrix_view, tuple(self._display_records_for_state(state))

    def _on_frame_search_requested(self) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        query = str(
            getattr(self, "frame_search_input", None).text() if hasattr(self, "frame_search_input") else ""
        ).strip()
        if not query:
            QMessageBox.information(self._view, self._t("dialog.info_title"), self._t("frame_search.empty"))
            return
        query_number = None
        digits = re.sub(r"\D+", "", query)
        if digits:
            try:
                query_number = int(digits)
            except Exception:
                query_number = None
        view, records = self._frame_search_view_and_records(state)
        matches: list[tuple[int, int, FrameRecord]] = []
        for index, record in enumerate(records):
            rank = self._frame_search_match_rank(record, query, query_number)
            if rank is not None:
                matches.append((int(rank), int(index), record))
        if not matches:
            QMessageBox.information(
                self._view, self._t("dialog.info_title"), self._t("frame_search.not_found", query=query)
            )
            return
        _rank, _index, record = min(matches, key=lambda item: (item[0], item[1]))
        selected = None
        if hasattr(view, "select_record_by_key"):
            selected = view.select_record_by_key(str(record.key), ensure_visible=True)
        if selected is None and view is not state.matrix_view and hasattr(state.matrix_view, "select_record_by_key"):
            state.matrix_view.select_record_by_key(str(record.key), ensure_visible=True)

    def _set_grid_inspection_calculation_scope(
        self, state: ExtendMatrixTabState, records: tuple[FrameRecord, ...]
    ) -> None:
        state.grid_inspection_calculation_record_keys = {
            str(record.key) for record in records if record is not None and str(record.key)
        }
        self._sync_action_buttons()

    def _connect_worker_profiling(self, worker: WorkerBase) -> None:
        dialog = getattr(self._view, "profiling_dialog", None)
        if dialog is not None:
            worker.profilingUpdated.connect(dialog.set_snapshot)
            worker.profilingFinished.connect(dialog.set_snapshot)
        for state in tuple(self._tab_states.values()):
            state.matrix_view.set_profiler(worker.profiler)
        for matrix_view in self._grid_inspection_views().values():
            if matrix_view is not None and hasattr(matrix_view, "set_profiler"):
                matrix_view.set_profiler(worker.profiler)

    def _start_build(self) -> None:
        explicit_model_specs = self._checked_model_specs()
        model_specs = self._effective_model_specs_for_build()
        required_model_count = self._required_model_count_for_build()
        building_from_base_only = (
            len(explicit_model_specs) <= 0
            and len(model_specs) == 1
            and self._original_folder is not None
            and Path(self._original_folder.path).exists()
        )
        if len(model_specs) < required_model_count and not building_from_base_only:
            message_key = (
                "errors.inter_model_model_count_required"
                if required_model_count > 1
                else "errors.active_model_required"
            )
            QMessageBox.warning(self._view, self._t("dialog.warning_title"), self._t(message_key))
            return
        self._close_all_details_dialogs()
        geometry_mode = geometry_mode_for_object_type(self._selected_object_type())
        mask_threshold, boundary_radius = self._selected_polygon_compare_values()
        options = BuildOptions(
            thumbnail_size=int(DEFAULT_CELL_SIZE),
            recursive=False,
            geometry_mode=geometry_mode,
            mask_threshold=float(mask_threshold),
            boundary_radius=int(boundary_radius),
            confidence_uncertainty_delta=self._selected_confidence_uncertainty_delta(),
            point_match_radius=float(self.point_match_radius_spin.value()),
            point_confidence_radius=int(self.point_confidence_radius_spin.value()),
            point_extraction_mode=str(self.point_extraction_mode_combo.currentData() or DEFAULT_POINT_EXTRACTION_MODE),
            polygon_confidence_summary=str(
                self.polygon_confidence_summary_combo.currentData() or DEFAULT_POLYGON_CONFIDENCE_SUMMARY
            ),
            comparison_pairs=self._selected_comparison_pairs(),
        )
        self._pending_build_snapshot = self._capture_view_snapshot()
        self._worker_kind = "build"
        self._worker_thread = QThread(self._view)
        self._worker = FrameIndexWorker(
            model_specs,
            options,
            self._original_folder,
            performance_config=self._view.performance_config,
        )
        self._connect_worker_profiling(self._worker)
        generation = self._begin_worker_request(state=None)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(
            lambda current, total, key, g=generation: self._on_build_progress(current, total, key, generation=g)
        )
        self._worker.finished.connect(lambda result, g=generation: self._on_build_finished(result, generation=g))
        self._worker.failed.connect(lambda message, g=generation: self._on_worker_failed(message, generation=g))
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)
        self._worker_thread.start()
        self._show_progress_bar(visible=True, format_text="Indexing frames...")
        self._sync_action_buttons()

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
        metric_key = str(state.metric_key or self.metric_combo.currentData() or DEFAULT_MATRIX_METRIC_KEY)
        request_signature = self._analytics_request_signature(state, metric_key)
        self._worker_kind = "analytics"
        self._active_compute_state = state
        self._worker_thread = QThread(self._view)
        self._worker = AnalyticsWorker(
            state.build_result,
            metric_key,
            self._excluded_record_keys_for_state(state),
            performance_config=self._view.performance_config,
        )
        self._connect_worker_profiling(self._worker)
        generation = self._begin_worker_request(state=state)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(
            lambda current, total, key, g=generation: self._on_build_progress(current, total, key, generation=g)
        )
        if hasattr(self._worker, "frameStateChanged"):
            self._worker.frameStateChanged.connect(
                lambda key, status, g=generation: self._on_frame_state_changed(key, status, generation=g)
            )
        self._worker.finished.connect(
            lambda result, g=generation, s=request_signature: self._on_analytics_finished(
                result, generation=g, request_signature=s
            )
        )
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
        if self._current_app_mode() == "grid_inspection":
            scope_records = self._grid_inspection_calculation_scope_records(state)
            self._start_compute_grid_inspection(state, selected_records=scope_records if scope_records else None)
            return
        if self._is_base_only_build_result(state.build_result):
            QMessageBox.information(
                self._view,
                self._t("dialog.info_title"),
                self._t("matrix.info.base_only_metrics_disabled"),
            )
            self._sync_action_buttons()
            return
        self._sync_current_analysis_context(state, auto_recompute=False)
        self._apply_pending_display_controls(state)
        self._start_compute_analytics(state=state, sync_context=False, apply_pending_controls=False)

    def _on_export_result_layer_requested(self) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        self._export_result_layer_jpgs(state, records=self._current_matrix_selected_records() or None)

    def _on_export_grid_check_bmps_requested(self) -> None:
        state = self._current_tab_state()
        if state is None or self._current_app_mode() != "grid_inspection":
            return
        selected_records = self._grid_inspection_export_selected_records(state)
        records_with_results = self._grid_inspection_records_with_results(state)
        self._export_grid_inspection_bmps(
            state,
            records_with_results,
            render_records=selected_records if selected_records else None,
        )

    def _request_cancel_build(self) -> None:
        if self._worker is None:
            return
        request_cancel = getattr(self._worker, "request_cancel", None)
        if callable(request_cancel):
            request_cancel()
        self._show_progress_bar(visible=True, format_text="Cancelling...")

    def _start_compute_grid_inspection(
        self,
        state: ExtendMatrixTabState,
        *,
        selected_records: tuple[FrameRecord, ...] | None = None,
    ) -> None:
        if not hasattr(self, "grid_inspection_matrix_view"):
            return
        excluded_keys = self._excluded_record_keys_for_state(state)
        partial_keys = {
            str(record.key) for record in (selected_records or ()) if record is not None and str(record.key)
        }
        source_records = (
            tuple(selected_records) if selected_records is not None else tuple(state.build_result.records or ())
        )
        records = tuple(
            record for record in source_records if str(getattr(record, "key", "") or "") not in excluded_keys
        )
        if selected_records is not None and not records:
            return
        state.matrix_score_view_mode = str(self.matrix_score_view_combo.currentData() or DEFAULT_MATRIX_SCORE_VIEW_MODE)
        next_config = self._grid_inspection_config_payload()
        grid_config = self._grid_damage_config_from_payload(next_config)
        model_id = self._grid_inspection_model_id_for_state(state)
        state.grid_inspection_model_id = model_id
        has_selected_sources = bool(
            model_id
            and any(
                bool((record.model_mask_paths or {}).get(str(model_id)))
                or bool((record.model_prob_paths or {}).get(str(model_id)))
                for record in records
            )
        )
        reference_selected = bool(str(getattr(state, "grid_inspection_reference_record_key", "") or ""))
        reference_profile = self._grid_inspection_reference_profile_for_state(state, grid_config)
        if reference_selected and reference_profile is None:
            self._sync_action_buttons()
            return
        state.grid_inspection_config_payload = next_config
        state.grid_inspection_results_ready = False
        state.grid_inspection_payload_by_key = {}
        state.grid_inspection_payloads_by_layer = {key: {} for key in self._grid_inspection_layer_keys()}
        self._sync_grid_inspection_layer_tabs(state)
        for view in self._grid_inspection_views().values():
            view.set_grid_inspection_payloads({}, enabled=True)
        self._refresh_grid_inspection_errors_panel(state)
        self._worker_kind = "grid_inspection"
        self._active_compute_state = state
        self._active_processing_keys = set()
        self._active_grid_inspection_partial_keys = set(partial_keys) if selected_records is not None else None
        state.processing_state_by_key.clear()
        state.matrix_view.set_processing_keys(set())
        for view in self._grid_inspection_views().values():
            view.set_processing_keys(set())
        self._worker_thread = QThread(self._view)
        if has_selected_sources and model_id:
            self._worker = PairedGridInspectionWorker(
                records,
                model_id,
                grid_config,
                reference_profile=reference_profile,
                performance_config=self._view.performance_config,
            )
        else:
            self._worker = GridInspectionWorker(
                records,
                replace(grid_config, cell_representation="binary"),
                model_id=model_id,
                reference_profile=None,
                performance_config=self._view.performance_config,
            )
        self._connect_worker_profiling(self._worker)
        generation = self._begin_worker_request(state=state)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(
            lambda current, total, key, g=generation: self._on_build_progress(current, total, key, generation=g)
        )
        if hasattr(self._worker, "frameStateChanged"):
            self._worker.frameStateChanged.connect(
                lambda key, status, g=generation: self._on_frame_state_changed(key, status, generation=g)
            )
        if hasattr(self._worker, "frameStatesChanged"):
            self._worker.frameStatesChanged.connect(
                lambda keys, status, g=generation: self._on_frame_states_changed(keys, status, generation=g)
            )
        if hasattr(self._worker, "partialResultsReady"):
            self._worker.partialResultsReady.connect(
                lambda payloads, g=generation: self._on_grid_inspection_batch(payloads, generation=g)
            )
        self._worker.finished.connect(
            lambda payloads, g=generation: self._on_grid_inspection_finished(payloads, generation=g)
        )
        self._worker.failed.connect(lambda message, g=generation: self._on_worker_failed(message, generation=g))
        self._worker.cancelled.connect(lambda g=generation: self._on_grid_inspection_cancelled(generation=g))
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker.cancelled.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)
        self._show_progress_bar(visible=True, format_text="Computing cell defects...")
        self._worker_thread.start()
        self._sync_action_buttons()

    def _cleanup_worker(self) -> None:
        active_state = self._active_compute_state
        if self._worker is not None:
            self._worker.deleteLater()
        if self._worker_thread is not None:
            self._worker_thread.deleteLater()
        if active_state is not None:
            active_state.matrix_view.set_processing_keys(set())
            active_state.processing_state_by_key.clear()
        for view in self._grid_inspection_views().values():
            view.set_processing_keys(set())
        self._worker = None
        self._worker_thread = None
        self._worker_kind = None
        self._active_compute_state = None
        self._active_request_generation = None
        self._active_processing_keys = set()
        self._active_grid_inspection_partial_keys = None
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
                return
        auto_compute_state = self._auto_compute_state_after_cleanup
        self._auto_compute_state_after_cleanup = None
        if auto_compute_state is not None and auto_compute_state.widget in self._tab_states:
            QTimer.singleShot(0, self._on_compute_requested)

    def _apply_grid_inspection_worker_payloads(
        self,
        state: ExtendMatrixTabState,
        payloads: dict[str, object],
        *,
        replace_all: bool,
    ) -> None:
        layers = (
            {key: {} for key in self._grid_inspection_layer_keys()}
            if replace_all
            else {
                key: dict(value)
                for key, value in (getattr(state, "grid_inspection_payloads_by_layer", {}) or {}).items()
                if key in self._grid_inspection_layer_keys()
            }
        )
        for layer_key in self._grid_inspection_layer_keys():
            layers.setdefault(layer_key, {})
        changed = {key: {} for key in self._grid_inspection_layer_keys()}
        for record_key, payload in payloads.items():
            if isinstance(payload, dict):
                for layer_key in self._grid_inspection_layer_keys():
                    result = payload.get(layer_key)
                    if result is not None:
                        layers[layer_key][str(record_key)] = result
                        changed[layer_key][str(record_key)] = result
            else:
                layers["binary"][str(record_key)] = payload
                changed["binary"][str(record_key)] = payload
        state.grid_inspection_payloads_by_layer = layers
        active_layer = str(getattr(state, "grid_inspection_layer", "confidence") or "confidence")
        state.grid_inspection_payload_by_key = dict(layers.get(active_layer, {}) or {})
        state.grid_inspection_results_ready = any(bool(value) for value in layers.values())
        self._sync_grid_inspection_layer_tabs(state)
        for layer_key, view in self._grid_inspection_views().items():
            if replace_all:
                view.set_grid_inspection_payloads(layers.get(layer_key, {}), enabled=True)
            elif changed.get(layer_key):
                view.update_grid_inspection_payloads(changed[layer_key])

    def _on_grid_inspection_finished(self, payloads: object, *, generation: int | None = None) -> None:
        if not self._is_active_request_generation(generation):
            return
        self._show_progress_bar(visible=False)
        state = self._active_compute_state or self._current_tab_state()
        if state is None or not hasattr(self, "grid_inspection_matrix_view"):
            return
        payload_map = {str(key): value for key, value in payloads.items()} if isinstance(payloads, dict) else {}
        self._apply_grid_inspection_worker_payloads(state, payload_map, replace_all=True)
        state.percentile_cache.clear()
        state.repeated_percentile_cache.clear()
        self._active_processing_keys = set()
        state.processing_state_by_key.clear()
        state.matrix_view.set_processing_keys(set())
        for view in self._grid_inspection_views().values():
            view.set_processing_keys(set())
            view.finalize_grid_inspection_payloads()
        if self._current_app_mode() == "grid_inspection":
            self.grid_inspection_matrix_view.viewport().update()
            QTimer.singleShot(50, lambda s=state: self._finish_grid_inspection_summary_refresh(s))
        self._sync_action_buttons()

    def _finish_grid_inspection_summary_refresh(self, state: ExtendMatrixTabState) -> None:
        if (
            state.widget not in self._tab_states
            or self._current_tab_state() is not state
            or self._current_app_mode() != "grid_inspection"
        ):
            return
        self._refresh_grid_inspection_errors_panel(state)
        self._schedule_metric_histogram_update(state)

    def _on_grid_inspection_batch(self, payloads: object, *, generation: int | None = None) -> None:
        if not self._is_active_request_generation(generation) or not isinstance(payloads, dict):
            return
        state = self._active_compute_state
        if state is None or not hasattr(self, "grid_inspection_matrix_view"):
            return
        batch = {str(key): value for key, value in payloads.items()}
        if not batch:
            return
        self._apply_grid_inspection_worker_payloads(state, batch, replace_all=False)

    def _on_grid_inspection_cancelled(self, *, generation: int | None = None) -> None:
        if not self._is_active_request_generation(generation):
            return
        self._show_progress_bar(visible=False)
        state = self._active_compute_state
        if state is not None:
            state.grid_inspection_results_ready = bool(state.grid_inspection_payload_by_key)
            state.processing_state_by_key.clear()
            state.matrix_view.set_processing_keys(set())
        self._active_processing_keys = set()
        for view in self._grid_inspection_views().values():
            view.set_processing_keys(set())
        self._sync_action_buttons()

    def _on_build_progress(self, current: int, total: int, key: str, *, generation: int | None = None) -> None:
        if not self._is_active_request_generation(generation):
            return
        self._active_progress_current = int(current)
        self._active_progress_total = int(total)
        self._active_progress_key = str(key or "")
        if self._active_compute_state is not None and not self._active_processing_keys:
            fallback_keys = {str(key)} if key else set()
            self._active_compute_state.matrix_view.set_processing_keys(fallback_keys)
            if self._worker_kind == "grid_inspection":
                for view in self._grid_inspection_views().values():
                    view.set_processing_keys(fallback_keys)
        format_text = self._progress_format_text(current, total, str(key or ""))
        self._show_progress_bar(visible=True, current=current, total=total, key=key, format_text=format_text)

    def _on_build_finished(self, result: BuildResult, *, generation: int | None = None) -> None:
        if not self._is_active_request_generation(generation):
            return
        snapshot = self._pending_build_snapshot or self._capture_view_snapshot()
        snapshot["confidence_model_id"] = (
            snapshot.get("confidence_model_id") or snapshot.get("metric_scope") or default_confidence_model_id(result)
        )
        self._sync_metric_controls(
            result,
            preferred_metric_key=str(snapshot.get("metric_key") or DEFAULT_MATRIX_METRIC_KEY),
            preferred_scope_key=str(snapshot.get("confidence_model_id") or ""),
        )
        snapshot["metric_scope"] = str(self.metric_scope_combo.currentData() or "")
        snapshot["confidence_model_id"] = str(self.metric_scope_combo.currentData() or "")
        snapshot["metric_key"] = str(
            self.metric_combo.currentData() or self._default_metric_key_for_state(None, result)
        )
        state = self._create_matrix_tab(result, snapshot)
        self._last_active_tab_state = state
        self._connect_histogram_cards(state)
        ok = self._apply_tab_visual_settings(state, reset_view=True, update_histograms=False)
        self._show_progress_bar(visible=False)
        if not ok:
            self._auto_compute_after_build = False
            self._auto_compute_state_after_cleanup = None
            state.widget.deleteLater()
            return
        profile_title = self._t(analysis_profile_definition(self._analysis_profile).title_key)
        run_time = datetime.now().strftime("%H:%M:%S")
        title = f"{profile_title} [{run_time}]"
        self._tab_states[state.widget] = state
        tab_index = self.matrix_tabs.addTab(state.widget, title)
        self.matrix_tabs.setCurrentIndex(tab_index)
        if hasattr(self, "run_history_list"):
            history_item = QListWidgetItem(f"{run_time} · {profile_title} · {len(result.model_specs)} models")
            history_item.setData(Qt.ItemDataRole.UserRole, state.widget)
            history_item.setToolTip(title)
            self.run_history_list.addItem(history_item)
            self.run_history_list.setCurrentItem(history_item)
            self.run_history_group.show()
        if result.records:
            state.matrix_view.select_record_by_key(result.records[0].key, ensure_visible=False)
            self._update_matrix_preview(state, result.records[0])
        if self._current_app_mode() == "grid_inspection":
            self._refresh_grid_inspection_mode_view()
        self._schedule_metric_histogram_update(state)
        self._sync_action_buttons()
        if self._auto_compute_after_build:
            self._auto_compute_after_build = False
            self._auto_compute_state_after_cleanup = state

    def _on_analytics_finished(
        self, result: BuildResult, *, generation: int | None = None, request_signature: tuple[object, ...] | None = None
    ) -> None:
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
        state.last_analytics_request_signature = request_signature or self._analytics_request_signature(
            state, state.metric_key
        )
        state.grid_inspection_results_ready = False
        state.grid_inspection_payload_by_key = {}
        state.grid_inspection_payloads_by_layer = {}
        self._invalidate_state_runtime_caches(state, clear_metric_results=True)
        self._sync_metric_controls(
            result,
            preferred_metric_key=result.selected_metric_key,
            preferred_scope_key=state.confidence_model_id or state.metric_scope,
            context_state=state,
        )
        state.confidence_model_id = self._selected_confidence_model_id(result)
        state.metric_scope = str(state.confidence_model_id or "")
        state.metric_key = str(
            self.metric_combo.currentData()
            or result.selected_metric_key
            or self._default_metric_key_for_state(state, result)
        )
        self._apply_metric_to_state(state, state.metric_key)
        if self._current_app_mode() == "grid_inspection":
            self._refresh_grid_inspection_mode_view()
        self._sync_action_buttons()

    def _on_worker_failed(self, message: str, *, generation: int | None = None) -> None:
        if not self._is_active_request_generation(generation):
            return
        self._show_progress_bar(visible=False)
        self._auto_compute_after_build = False
        self._auto_compute_state_after_cleanup = None
        if self._active_compute_state is not None:
            self._active_compute_state.matrix_view.set_processing_keys(set())
            self._active_compute_state.processing_state_by_key.clear()
        if message and "cancel" in message.lower():
            self._sync_action_buttons()
            return
        QMessageBox.critical(
            self._view, self._t("dialog.warning_title"), message or self._t("errors.background_failed")
        )

    def _apply_metric_to_state(self, state: ExtendMatrixTabState, metric_key: str) -> None:
        available = set(state.build_result.available_metric_keys or ())
        can_request_dynamic_metric = self._is_dynamic_pair_metric_key(metric_key) and bool(
            getattr(state.build_result, "scores_computed", False)
        )
        if available and metric_key not in available and not can_request_dynamic_metric:
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
            updated_records.append(
                replace(
                    record,
                    score=float(display),
                    absolute_score=float(absolute),
                    relative_score=float(relative),
                    score_ready=True,
                )
            )
        percentile_map = compute_metric_percentiles(updated_records, metric_key)
        updated_records = [
            replace(record, score_percentile=float(percentile_map.get(record.key, 0.0))) for record in updated_records
        ]
        state.build_result = replace(
            state.build_result,
            records=tuple(updated_records),
            min_score=min((record.score for record in updated_records), default=0.0),
            max_score=max((record.score for record in updated_records), default=0.0),
            min_absolute_score=min_absolute,
            max_absolute_score=max_absolute,
            selected_metric_key=metric_key,
        )
        state.metric_result_cache[metric_key] = state.build_result
        self._apply_tab_visual_settings(state, reset_view=False)
        state.grid_inspection_results_ready = False
        state.grid_inspection_payload_by_key = {}
        state.grid_inspection_payloads_by_layer = {}

    def _on_metric_scope_changed(self, *_args) -> None:
        state = self._current_tab_state()
        preferred_scope = str(self.metric_scope_combo.currentData() or "")
        build_result = None if state is None else state.build_result
        self._sync_metric_controls(build_result, preferred_scope_key=preferred_scope, context_state=state)
        if state is None:
            return
        self.metric_combo.setToolTip(
            self._metric_hint_fallback(
                str(self.metric_combo.currentData() or DEFAULT_MATRIX_METRIC_KEY), state.build_result
            )
        )
        self._sync_action_buttons()

    def _on_analysis_mode_changed(self, *_args) -> None:
        state = self._current_tab_state()
        self._apply_global_analysis_context_to_all_states()
        self._sync_mode_controls(state, None if state is None else state.build_result)
        if state is not None:
            self._sync_metric_controls(
                state.build_result,
                preferred_metric_key=state.metric_key,
                preferred_scope_key=state.confidence_model_id or state.metric_scope,
                context_state=state,
            )
        self._sync_action_buttons()

    def _on_comparison_target_changed(self, *_args) -> None:
        state = self._current_tab_state()
        self._sync_mode_controls(state, None if state is None else state.build_result)
        self._refresh_pair_matrix()
        if state is None:
            self._sync_action_buttons()
            return
        self._apply_comparison_target_to_states()
        preferred_metric = self._default_metric_key_for_comparison_target(state.build_result) or state.metric_key
        self._sync_metric_controls(
            state.build_result,
            preferred_metric_key=preferred_metric,
            preferred_scope_key=state.confidence_model_id or state.metric_scope,
            context_state=state,
        )
        metric_key = str(self.metric_combo.currentData() or preferred_metric or DEFAULT_MATRIX_METRIC_KEY)
        self._apply_metric_to_state(state, metric_key)
        self._sync_action_buttons()

    def _on_object_type_changed(self, *_args) -> None:
        state = self._current_tab_state()
        self._apply_global_analysis_context_to_all_states()
        self._sync_mode_controls(state, None if state is None else state.build_result)
        if state is not None:
            self._sync_metric_controls(
                state.build_result,
                preferred_metric_key=state.metric_key,
                preferred_scope_key=state.confidence_model_id or state.metric_scope,
                context_state=state,
            )
        self._sync_action_buttons()

    def _on_polygon_compare_profile_changed(self, *_args) -> None:
        self._apply_polygon_compare_profile(self._selected_polygon_compare_profile())
        self._sync_action_buttons()

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
        state = self._current_tab_state()
        if state is not None:
            state.matrix_score_view_mode = str(
                self.matrix_score_view_combo.currentData() or DEFAULT_MATRIX_SCORE_VIEW_MODE
            )
            if self._current_app_mode() == "grid_inspection":
                self._refresh_grid_inspection_mode_view()
        self._sync_action_buttons()

    def _on_matrix_gradient_changed(self, *_args) -> None:
        gradient_name = str(self.matrix_gradient_combo.currentData() or DEFAULT_GRADIENT_NAME)
        state = self._current_tab_state()
        if state is not None:
            state.gradient_name = gradient_name
            state.matrix_view.set_gradient_preset(gradient_name)
        for view in self._grid_inspection_views().values():
            view.set_gradient_preset(gradient_name)
        self._sync_action_buttons()

    def _on_grid_inspection_tuning_changed(self, *_args) -> None:
        # Tuning controls are staged values. They are applied only when the user starts
        # the next grid-inspection calculation.
        self._sync_action_buttons()

    def _on_matrix_visual_parameter_changed(self, *_args) -> None:
        state = self._current_tab_state()
        self._sync_mode_controls(state, None if state is None else state.build_result)
        self._sync_action_buttons()

    def _on_current_tab_changed(self, _index: int) -> None:
        state = self._current_tab_state()
        if state is None:
            self._sync_action_buttons()
            if self._current_app_mode() == "grid_inspection":
                self._refresh_grid_inspection_mode_view()
            return
        self._last_active_tab_state = state
        if hasattr(self, "run_history_list"):
            for row in range(self.run_history_list.count()):
                item = self.run_history_list.item(row)
                if item.data(Qt.ItemDataRole.UserRole) is state.widget:
                    self.run_history_list.setCurrentItem(item)
                    break
        self._set_ui_context_from_state(state)
        self._apply_global_analysis_context_to_state(state)
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
        state.metric_key = str(
            self.metric_combo.currentData()
            or state.metric_key
            or self._default_metric_key_for_state(state, state.build_result)
        )
        self.metric_combo.setToolTip(self._metric_hint_fallback(state.metric_key, state.build_result))
        if self._metric_value_missing_for_build_result(state.build_result, state.metric_key):
            self._update_matrix_preview(state)
            if self._worker is None and bool(getattr(state.build_result, "scores_computed", False)):
                self._start_compute_analytics(state=state, sync_context=False)
            elif self._worker is not None and bool(getattr(state.build_result, "scores_computed", False)):
                self._deferred_analytics_restart = (state, False)
            self._sync_action_buttons()
            return
        if (
            not bool(getattr(state.build_result, "scores_computed", False))
            or str(getattr(state.build_result, "selected_metric_key", "")) != state.metric_key
        ):
            self._apply_metric_to_state(state, state.metric_key)
            self._sync_action_buttons()
            return
        self._update_matrix_preview(state)
        if self._current_app_mode() == "grid_inspection":
            self._refresh_grid_inspection_mode_view()
        self._sync_action_buttons()

    def _close_matrix_tab(self, index: int) -> None:
        widget = self.matrix_tabs.widget(index)
        if widget is None:
            return
        removed_state = self._tab_states.pop(widget, None)
        if removed_state is not None and self._last_active_tab_state is removed_state:
            self._last_active_tab_state = next(iter(self._tab_states.values()), None)
        self.matrix_tabs.removeTab(index)
        if hasattr(self, "run_history_list"):
            for row in range(self.run_history_list.count() - 1, -1, -1):
                item = self.run_history_list.item(row)
                if item.data(Qt.ItemDataRole.UserRole) is widget:
                    self.run_history_list.takeItem(row)
            self.run_history_group.setVisible(self.run_history_list.count() > 0)
        widget.deleteLater()
        if not self._tab_states and hasattr(self._view, "show_empty_matrix_state"):
            self._view.show_empty_matrix_state()
        self._sync_action_buttons()

    def _on_run_history_selected(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        widget = item.data(Qt.ItemDataRole.UserRole)
        index = self.matrix_tabs.indexOf(widget)
        if index >= 0:
            self.matrix_tabs.setCurrentIndex(index)

    def _on_matrix_overview_changed(
        self,
        state: ExtendMatrixTabState,
        image,
        visible_rect,
        selected_position,
        selected_blink_on,
        processing_positions,
        reference_position,
    ) -> None:
        state.mini_map.set_overview(
            image, visible_rect, selected_position, selected_blink_on, processing_positions, reference_position
        )

    def _on_record_selected(self, state: ExtendMatrixTabState, record: FrameRecord | None) -> None:
        if self._current_tab_state() is state:
            self._update_matrix_preview(state, record)
            self._sync_action_buttons()

    def _forget_details_dialog(self, dialog: ExtendFrameDetailsDialog) -> None:
        self._details_dialogs = [opened for opened in self._details_dialogs if opened is not dialog]

    def _close_all_details_dialogs(self) -> None:
        for dialog in list(self._details_dialogs):
            dialog.close()
        self._details_dialogs.clear()

    @staticmethod
    def _grid_inspection_source_path_for_record(record: FrameRecord) -> str:
        model_masks = getattr(record, "model_mask_paths", {}) or {}
        if model_masks:
            return str(next(iter(model_masks.values())) or "")
        return str(
            getattr(record, "first_path", "")
            or getattr(record, "base_path", "")
            or getattr(record, "original_path", "")
            or ""
        )

    @staticmethod
    def _grid_inspection_layer_source_path_for_record(
        record: FrameRecord,
        model_id: str | None,
        layer_key: str,
    ) -> str:
        model_key = str(model_id or "")
        if str(layer_key) == "confidence" and model_key:
            path_text = str((getattr(record, "model_prob_paths", {}) or {}).get(model_key) or "")
            if path_text:
                return path_text
        if model_key:
            path_text = str((getattr(record, "model_mask_paths", {}) or {}).get(model_key) or "")
            if path_text:
                return path_text
        return KarakalPresenter._grid_inspection_source_path_for_record(record)

    @staticmethod
    def _grid_inspection_display_source_path_for_record(record: FrameRecord) -> str:
        source_path = str(getattr(record, "original_path", "") or getattr(record, "base_path", "") or "")
        if source_path:
            return source_path
        return KarakalPresenter._grid_inspection_source_path_for_record(record)

    def _open_record_details(
        self, record: FrameRecord, state: ExtendMatrixTabState, grid_focus: dict[str, object] | None = None
    ) -> None:
        session_view_state = dict(self._details_view_payload)
        is_grid_inspection_details = self._current_app_mode() == "grid_inspection"
        preferred_model_id = self._preferred_details_model_id_for_state(state, session_view_state=session_view_state)
        session_view_state["preferred_model_id"] = preferred_model_id
        session_view_state["analysis_mode"] = str(state.analysis_mode or INTER_MODEL_ANALYSIS_MODE)
        session_view_state["result_kind"] = (
            "grid_cell_defects" if is_grid_inspection_details else self._default_details_result_kind_for_state(state)
        )
        session_view_state["layer_view"] = self._default_details_layer_view_for_state(state)
        session_view_state["comparison_mode"] = self._default_details_comparison_mode_for_state(state)
        session_view_state["grayscale_diff"] = self._default_details_grayscale_diff_for_state(state)
        if is_grid_inspection_details:
            session_view_state["grid_damage_config"] = dict(
                getattr(state, "grid_inspection_config_payload", {}) or self._grid_inspection_config_payload()
            )
        grid_inspection_result = None
        grid_inspection_source_path = None
        if is_grid_inspection_details and bool(getattr(state, "grid_inspection_results_ready", False)):
            grid_inspection_result = (getattr(state, "grid_inspection_payload_by_key", {}) or {}).get(
                str(getattr(record, "key", "") or "")
            )
            grid_inspection_source_path = self._grid_inspection_display_source_path_for_record(record)
        dialog = ExtendFrameDetailsDialog(
            record=record,
            build_result=state.build_result,
            preferred_metric_key=state.metric_key,
            session_view_state=session_view_state,
            on_view_state_changed=self._store_details_view_payload,
            export_folder=self._export_folder,
            allowed_result_kinds=("grid_cell_defects",) if is_grid_inspection_details else None,
            grid_inspection_result=grid_inspection_result,
            grid_inspection_source_path=grid_inspection_source_path,
            performance_config=self._view.performance_config,
            parent=None,
        )
        dialog.setModal(False)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.destroyed.connect(lambda *_args, dialog=dialog: self._forget_details_dialog(dialog))
        self._details_dialogs.append(dialog)
        dialog.show()
        if grid_focus:
            focus = getattr(dialog, "focus_grid_cell_defect", None)
            if callable(focus):
                focus(grid_focus)
        dialog.raise_()
        dialog.activateWindow()

    def _on_matrix_context_menu(
        self, state: ExtendMatrixTabState, record: FrameRecord | None, global_pos, matrix_view=None
    ) -> None:
        source_view = matrix_view or state.matrix_view
        selected_records = tuple()
        if hasattr(source_view, "selected_records"):
            try:
                selected_records = tuple(source_view.selected_records())
            except Exception:
                selected_records = tuple()
        menu = QMenu(self._view)
        single_selected_record = selected_records[0] if len(selected_records) == 1 else None
        open_action = None
        if single_selected_record is not None:
            open_action = menu.addAction(self._t("context.open_details"))
        exclude_source = tuple(selected_records) if selected_records else ((record,) if record is not None else tuple())
        saved_validation_mask = self._saved_validation_mask_record_keys()
        is_grid_inspection_context = source_view is getattr(self, "grid_inspection_matrix_view", None)
        export_menu = menu.addMenu(self._t("context.export_menu"))
        export_action = export_menu.addAction(self._t("context.export_frame_assets"))
        export_action.setEnabled(record is not None)
        export_selected_action = export_menu.addAction(
            self._t("context.export_selected_frame_assets", count=len(selected_records))
        )
        export_selected_action.setEnabled(bool(selected_records))
        export_layer_action = export_menu.addAction(self._t("context.export_result_layer_jpgs"))
        export_layer_action.setEnabled(bool(getattr(state.build_result, "records", ())))
        export_grid_bmp_selected_action = None
        export_grid_bmp_all_action = None
        if is_grid_inspection_context:
            export_menu.addSeparator()
            payload_keys = {str(key) for key in (getattr(state, "grid_inspection_payload_by_key", {}) or {}).keys()}
            selected_result_keys = {str(getattr(item, "key", "") or "") for item in exclude_source if item is not None}
            export_grid_bmp_selected_action = export_menu.addAction(
                self._t("context.export_selected_grid_check_bmps", count=len(exclude_source))
            )
            export_grid_bmp_selected_action.setEnabled(bool(selected_result_keys & payload_keys))
            export_grid_bmp_all_action = export_menu.addAction(self._t("context.export_all_grid_check_bmps"))
            export_grid_bmp_all_action.setEnabled(bool(payload_keys))
        validation_menu = menu.addMenu(
            self._t("context.grid_inspection_menu")
            if is_grid_inspection_context
            else self._t("context.validation_menu")
        )
        exclude_action = validation_menu.addAction(
            self._t("context.exclude_selected_from_grid_inspection", count=len(exclude_source))
            if is_grid_inspection_context
            else self._t("context.exclude_selected_from_validation", count=len(exclude_source))
        )
        exclude_action.setEnabled(bool(exclude_source))
        restore_action = validation_menu.addAction(
            self._t("context.restore_selected_grid_inspection", count=len(exclude_source))
            if is_grid_inspection_context
            else self._t("context.restore_selected_validation", count=len(exclude_source))
        )
        restore_action.setEnabled(bool(exclude_source))
        compute_selected_action = None
        clear_calculation_scope_action = None
        if is_grid_inspection_context:
            compute_selected_action = validation_menu.addAction(
                self._t("context.compute_selected_grid_inspection", count=len(exclude_source))
            )
            compute_selected_action.setEnabled(bool(exclude_source))
            clear_calculation_scope_action = validation_menu.addAction(self._t("context.compute_all_grid_inspection"))
            clear_calculation_scope_action.setEnabled(
                bool(getattr(state, "grid_inspection_calculation_record_keys", set()))
            )
        validation_menu.addSeparator()
        save_mask_action = validation_menu.addAction(self._t("context.save_validation_mask"))
        save_mask_action.setEnabled(bool(state.excluded_record_keys))
        apply_mask_action = validation_menu.addAction(self._t("context.apply_validation_mask"))
        apply_mask_action.setEnabled(bool(saved_validation_mask))
        clear_all_exclusions_action = validation_menu.addAction(
            self._t("context.clear_all_grid_inspection_exclusions")
            if is_grid_inspection_context
            else self._t("context.clear_all_validation_exclusions")
        )
        clear_all_exclusions_action.setEnabled(bool(state.excluded_record_keys))
        selected_action = menu.exec(global_pos)
        if selected_action is None:
            return
        if open_action is not None and selected_action is open_action and single_selected_record is not None:
            self._open_record_details(single_selected_record, state)
            return
        if selected_action is export_action and record is not None:
            self._export_record_assets(state, record)
            return
        if selected_action is export_selected_action and selected_records:
            self._export_records_assets(state, selected_records)
            return
        if selected_action is export_layer_action:
            self._export_result_layer_jpgs(state)
            return
        if export_grid_bmp_selected_action is not None and selected_action is export_grid_bmp_selected_action:
            self._export_grid_inspection_bmps(
                state,
                self._grid_inspection_records_with_results(state),
                render_records=tuple(exclude_source),
            )
            return
        if export_grid_bmp_all_action is not None and selected_action is export_grid_bmp_all_action:
            self._export_grid_inspection_bmps(state, self._grid_inspection_records_with_results(state))
            return
        if selected_action is exclude_action and exclude_source:
            self._set_validation_exclusions(state, exclude_source, exclude=True)
            return
        if selected_action is restore_action and exclude_source:
            self._set_validation_exclusions(state, exclude_source, exclude=False)
            return
        if compute_selected_action is not None and selected_action is compute_selected_action and exclude_source:
            self._set_grid_inspection_calculation_scope(state, tuple(exclude_source))
            return
        if clear_calculation_scope_action is not None and selected_action is clear_calculation_scope_action:
            self._set_grid_inspection_calculation_scope(state, tuple())
            return
        if selected_action is save_mask_action:
            self._save_validation_mask(state)
            return
        if selected_action is apply_mask_action:
            self._apply_validation_mask(state)
            return
        if selected_action is clear_all_exclusions_action:
            self._set_validation_exclusions(state, tuple(), clear_all=True)

    def _on_grid_inspection_context_menu(self, record: FrameRecord | None, global_pos) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        self._on_matrix_context_menu(
            state, record, global_pos, matrix_view=getattr(self, "grid_inspection_matrix_view", None)
        )

    def _set_validation_exclusions(
        self,
        state: ExtendMatrixTabState,
        records: tuple[FrameRecord, ...],
        *,
        exclude: bool = False,
        clear_all: bool = False,
        replace_current: bool = False,
    ) -> None:
        previous = set(self._excluded_record_keys_for_state(state))
        current = set(previous)
        if clear_all or replace_current:
            current.clear()
        else:
            keys = {str(record.key) for record in records if record is not None}
            if exclude:
                current.update(keys)
            else:
                current.difference_update(keys)
        state.excluded_record_keys = current
        state.matrix_view.set_excluded_record_keys(set(current))
        if hasattr(self, "grid_inspection_matrix_view"):
            self.grid_inspection_matrix_view.set_excluded_record_keys(set(current))
        if current != previous:
            self._invalidate_state_runtime_caches(state, clear_metric_results=True)
            state.last_analytics_request_signature = None
            state.build_result = replace(state.build_result, scores_computed=False)
            state.grid_inspection_results_ready = False
            state.grid_inspection_payload_by_key = {}
            state.grid_inspection_payloads_by_layer = {}
            if hasattr(self, "grid_inspection_matrix_view"):
                self.grid_inspection_matrix_view.set_grid_inspection_payloads({}, enabled=True)
            self._refresh_grid_inspection_errors_panel(state)
        self._update_matrix_preview(state, state.matrix_view.current_record())
        self._sync_action_buttons()

    def _saved_validation_mask_record_keys(self) -> set[str]:
        payload = self._saved_validation_mask_payload if isinstance(self._saved_validation_mask_payload, dict) else {}
        keys = payload.get("excluded_record_keys") if isinstance(payload, dict) else None
        return {str(key) for key in (keys or ()) if str(key)}

    def _save_validation_mask(self, state: ExtendMatrixTabState) -> None:
        keys = {str(key) for key in self._excluded_record_keys_for_state(state) if str(key)}
        if not keys:
            return
        self._saved_validation_mask_payload = {
            "excluded_record_keys": tuple(sorted(keys)),
            "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "source_build": str(getattr(state.build_result, "name", "") or ""),
        }
        self._settings_service.save_validation_mask_payload(self._saved_validation_mask_payload)
        self._settings_service.sync()
        QMessageBox.information(
            self._view,
            self._t("dialog.info_title"),
            self._t("message.validation_mask_saved", count=len(keys)),
        )

    def _apply_validation_mask(self, state: ExtendMatrixTabState) -> None:
        saved_keys = self._saved_validation_mask_record_keys()
        if not saved_keys:
            return
        current_keys = {str(record.key) for record in getattr(state.build_result, "records", ()) if str(record.key)}
        keys = saved_keys & current_keys if current_keys else saved_keys
        records = tuple(record for record in getattr(state.build_result, "records", ()) if str(record.key) in keys)
        self._set_validation_exclusions(state, records, exclude=True, replace_current=True)
        QMessageBox.information(
            self._view,
            self._t("dialog.info_title"),
            self._t("message.validation_mask_applied", count=len(keys)),
        )

    def _ensure_export_folder(self) -> Path | None:
        if self._export_folder is None:
            folder = QFileDialog.getExistingDirectory(self._view, self._t("dialog.select_export_folder"))
            if not folder:
                return None
            self._export_folder = Path(folder)
            self._update_source_labels()
            self._sync_action_buttons()
        return self._export_folder

    def _export_record_assets(self, state: ExtendMatrixTabState, record: FrameRecord) -> None:
        export_folder = self._ensure_export_folder()
        if export_folder is None:
            return
        try:
            result = export_record_assets(state.build_result, record, export_folder, write_manifest=False)
        except Exception as error:
            QMessageBox.warning(self._view, self._t("dialog.warning_title"), str(error))
            return
        count = int(result.get("exported_count", 0))
        QMessageBox.information(
            self._view,
            self._t("dialog.info_title"),
            self._t("message.export_frame_done", count=count, folder=str(export_folder)),
        )

    def _export_records_assets(self, state: ExtendMatrixTabState, records: tuple[FrameRecord, ...]) -> None:
        export_folder = self._ensure_export_folder()
        if export_folder is None:
            return
        exported_files = 0
        exported_records = 0
        errors: list[str] = []
        for record in records:
            try:
                result = export_record_assets(state.build_result, record, export_folder, write_manifest=False)
            except Exception as error:
                errors.append(f"{record.display_name}: {error}")
                continue
            exported_files += int(result.get("exported_count", 0))
            exported_records += 1
        if errors:
            QMessageBox.warning(
                self._view,
                self._t("dialog.warning_title"),
                "\n".join(errors[:10]),
            )
            return
        QMessageBox.information(
            self._view,
            self._t("dialog.info_title"),
            self._t(
                "message.export_frames_done",
                frame_count=exported_records,
                file_count=exported_files,
                folder=str(export_folder),
            ),
        )

    @staticmethod
    def _record_source_asset_path(record: FrameRecord) -> Path | None:
        for value in (
            getattr(record, "original_path", ""),
            getattr(record, "base_path", ""),
            getattr(record, "first_path", ""),
        ):
            if not value:
                continue
            path = Path(str(value))
            if path.is_file():
                return path
        return None

    def _copy_percentile_source_asset(
        self,
        record: FrameRecord,
        sources_dir: Path,
        used_names: set[str],
    ) -> dict[str, str] | None:
        source_path = self._record_source_asset_path(record)
        if source_path is None:
            return None
        sources_dir.mkdir(parents=True, exist_ok=True)
        candidate_name = source_path.name
        if candidate_name in used_names:
            key_fragment = self._safe_export_fragment(getattr(record, "key", ""), fallback="frame")
            candidate_name = f"{source_path.stem}_{key_fragment}{source_path.suffix}"
        base_name = candidate_name
        counter = 2
        while candidate_name in used_names:
            candidate_name = f"{Path(base_name).stem}_{counter}{source_path.suffix}"
            counter += 1
        used_names.add(candidate_name)
        target_path = sources_dir / candidate_name
        shutil.copy2(source_path, target_path)
        return {
            "record_key": str(getattr(record, "key", "")),
            "source": str(source_path),
            "destination": str(target_path),
        }

    def _percentile_export_folder(
        self, state: ExtendMatrixTabState, metric_key: str, bin_index: int, export_root: Path
    ) -> Path:
        labels = PERCENTILE_BAND_TITLES
        band_label = labels[int(bin_index)] if 0 <= int(bin_index) < len(labels) else f"P{int(bin_index)}"
        metric_label = self._metric_label(str(metric_key), state.build_result)
        metric_fragment = self._safe_export_fragment(metric_label, fallback="metric")
        band_fragment = self._safe_export_fragment(band_label, fallback="percentile")
        return export_root / f"percentile_{metric_fragment}_{band_fragment}"

    def _export_percentile_records_assets(
        self,
        state: ExtendMatrixTabState,
        records: tuple[FrameRecord, ...],
        metric_key: str,
        bin_index: int,
    ) -> None:
        export_root = self._ensure_export_folder()
        if export_root is None:
            return
        destination = self._percentile_export_folder(state, metric_key, bin_index, export_root)
        destination.mkdir(parents=True, exist_ok=True)
        sources_dir = destination / "sources"
        used_source_names: set[str] = set()
        exported_files = 0
        exported_records = 0
        asset_exports: list[dict[str, str]] = []
        source_exports: list[dict[str, str]] = []
        errors: list[str] = []
        total = len(records)
        progress = QProgressDialog(
            self._t("message.export_percentile_progress", current=0, total=total, frame=""),
            self._t("common.cancel"),
            0,
            max(1, total),
            self._view,
        )
        progress.setWindowTitle(self._t("dialog.info_title"))
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(300)
        progress.setValue(0)
        QApplication.processEvents()
        for index, record in enumerate(records, start=1):
            progress.setLabelText(
                self._t(
                    "message.export_percentile_progress",
                    current=index,
                    total=total,
                    frame=str(getattr(record, "display_name", "") or getattr(record, "key", "")),
                )
            )
            QApplication.processEvents()
            if progress.wasCanceled():
                break
            try:
                result = export_record_assets(state.build_result, record, destination, write_manifest=False)
                source_result = self._copy_percentile_source_asset(record, sources_dir, used_source_names)
            except Exception as error:
                errors.append(f"{getattr(record, 'display_name', record.key)}: {error}")
                continue
            exported_files += int(result.get("exported_count", 0))
            exported_records += 1
            for file_entry in result.get("files", ()) or ():
                if isinstance(file_entry, dict):
                    normalized_entry = {str(key): str(value) for key, value in file_entry.items()}
                    normalized_entry.setdefault("record_key", str(getattr(record, "key", "")))
                    asset_exports.append(normalized_entry)
            if source_result is not None:
                source_exports.append(source_result)
            progress.setValue(index)
        progress.close()
        low_bound, high_bound = self._percentile_bin_bounds(int(bin_index))
        manifest = {
            "metric_key": str(metric_key),
            "metric_label": self._metric_label(str(metric_key), state.build_result),
            "percentile_bin_index": int(bin_index),
            "percentile_range": [float(low_bound), float(high_bound)],
            "record_count": int(exported_records),
            "records": [str(getattr(record, "key", "")) for record in records],
            "exported_files": int(exported_files),
            "files": asset_exports,
            "sources": source_exports,
        }
        manifest_path = destination / "percentile_export_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if errors:
            QMessageBox.warning(
                self._view,
                self._t("dialog.warning_title"),
                "\n".join(errors[:10]),
            )
            return
        QMessageBox.information(
            self._view,
            self._t("dialog.info_title"),
            self._t(
                "message.export_percentile_done",
                frame_count=exported_records,
                file_count=exported_files,
                folder=str(destination),
            ),
        )

    def _select_result_layer_exports(self, choices: tuple[dict[str, str], ...]) -> tuple[dict[str, str], ...]:
        dialog = QDialog(self._view)
        dialog.setWindowTitle(self._t("dialog.select_result_layers"))
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(self._t("dialog.result_layers_prompt"), dialog))
        layer_list = QListWidget(dialog)
        layer_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for choice in choices:
            title_key = str(choice.get("title_key") or "")
            title = self._t(title_key) if title_key else str(choice.get("title") or choice.get("key") or "")
            group = str(choice.get("group") or "")
            label = title if group == "detail" else (f"{title} [{group}]" if group else title)
            item = QListWidgetItem(label, layer_list)
            item.setData(Qt.ItemDataRole.UserRole, dict(choice))
        if layer_list.count() > 0:
            layer_list.item(0).setSelected(True)
        layout.addWidget(layer_list)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return tuple()
        selected: list[dict[str, str]] = []
        for item in layer_list.selectedItems():
            data = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(data, dict):
                selected.append(dict(data))
        return tuple(selected)

    def _export_result_layer_jpgs(
        self, state: ExtendMatrixTabState, *, records: tuple[FrameRecord, ...] | None = None
    ) -> None:
        export_folder = self._ensure_export_folder()
        if export_folder is None:
            return
        export_build_result = state.build_result
        if records is not None:
            selected_keys = {str(getattr(record, "key", "") or "") for record in records if record is not None}
            selected_records = tuple(
                record
                for record in (getattr(state.build_result, "records", ()) or ())
                if str(getattr(record, "key", "") or "") in selected_keys
            )
            if selected_records:
                export_build_result = replace(state.build_result, records=selected_records)
        try:
            choices = available_result_layer_exports(export_build_result)
        except Exception as error:
            QMessageBox.warning(self._view, self._t("dialog.warning_title"), str(error))
            return
        if not choices:
            QMessageBox.warning(
                self._view,
                self._t("dialog.warning_title"),
                self._t("message.no_result_layers"),
            )
            return
        selected_choices = self._select_result_layer_exports(tuple(choices))
        if not selected_choices:
            return
        color_presets: list[tuple[str, tuple[int, int, int] | None]] = [
            (self._t("layer_color_preset.red"), (255, 64, 64)),
            (self._t("layer_color_preset.cyan"), (0, 229, 255)),
            (self._t("layer_color_preset.white"), (255, 255, 255)),
            (self._t("layer_color_preset.custom"), None),
        ]
        preset_labels = [label for label, _rgb in color_presets]
        preset_label, preset_accepted = QInputDialog.getItem(
            self._view,
            self._t("dialog.select_layer_color_preset"),
            self._t("dialog.layer_color_preset_prompt"),
            preset_labels,
            0,
            False,
        )
        if not preset_accepted or not preset_label:
            return
        selected_rgb = next((rgb for label, rgb in color_presets if label == preset_label), None)
        if selected_rgb is None:
            color = QColorDialog.getColor(
                QColor(255, 64, 64),
                self._view,
                self._t("dialog.select_layer_color"),
            )
            if not color.isValid():
                return
            selected_rgb = (int(color.red()), int(color.green()), int(color.blue()))
        records_count = len(tuple(getattr(export_build_result, "records", ()) or ()))
        total_units = max(1, records_count * len(selected_choices))
        progress = QProgressDialog(
            self._t("message.export_result_layer_progress", current=0, total=total_units, frame=""),
            self._t("common.cancel"),
            0,
            total_units,
            self._view,
        )
        progress.setWindowTitle(self._t("context.export_result_layer_jpgs"))
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        last_progress_update = 0.0

        def on_progress(current: int, total: int, frame_name: str) -> None:
            nonlocal last_progress_update
            now = perf_counter()
            if int(current) < int(total) and now - last_progress_update < 0.15:
                return
            last_progress_update = now
            progress.setMaximum(max(1, int(total)))
            progress.setValue(max(0, min(int(current), max(1, int(total)))))
            progress.setLabelText(
                self._t(
                    "message.export_result_layer_progress",
                    current=int(current),
                    total=int(total),
                    frame=str(frame_name or ""),
                )
            )
            QApplication.processEvents()

        try:
            result = (
                export_result_layer_jpgs(
                    export_build_result,
                    export_folder,
                    layer_key=str(selected_choices[0].get("key") or ""),
                    map_color=selected_rgb,
                    max_workers=int(getattr(export_build_result.options, "max_workers", 0) or 0) or None,
                    progress_callback=on_progress,
                    cancel_check=progress.wasCanceled,
                )
                if len(selected_choices) == 1
                else export_result_layers_jpgs(
                    export_build_result,
                    export_folder,
                    layer_keys=tuple(str(choice.get("key") or "") for choice in selected_choices),
                    map_color=selected_rgb,
                    max_workers=int(getattr(export_build_result.options, "max_workers", 0) or 0) or None,
                    progress_callback=on_progress,
                    cancel_check=progress.wasCanceled,
                )
            )
        except Exception as error:
            progress.close()
            QMessageBox.warning(self._view, self._t("dialog.warning_title"), str(error))
            return
        progress.close()
        count = int(result.get("exported_count", 0))
        skipped = int(result.get("skipped_count", 0))
        destination = str(result.get("destination") or export_folder)
        if count <= 0:
            errors = tuple(result.get("errors", ()) or ())
            message = "\n".join(str(item) for item in errors[:10]) if errors else self._t("message.no_result_layers")
            QMessageBox.warning(self._view, self._t("dialog.warning_title"), message)
            return
        QMessageBox.information(
            self._view,
            self._t("dialog.info_title"),
            self._t(
                "message.export_result_layer_done",
                count=count,
                skipped=skipped,
                layer=str(
                    result.get("layer_title")
                    or self._t("message.export_result_layers_count", count=len(selected_choices))
                ),
                folder=destination,
            ),
        )

    def _grid_inspection_records_with_results(self, state: ExtendMatrixTabState) -> tuple[FrameRecord, ...]:
        payload_keys = {str(key) for key in (getattr(state, "grid_inspection_payload_by_key", {}) or {}).keys()}
        if not payload_keys:
            return tuple()
        return tuple(
            record
            for record in (getattr(state.build_result, "records", ()) or ())
            if str(getattr(record, "key", "") or "") in payload_keys
        )

    def _select_grid_check_export_format(self) -> str | None:
        options = (
            (self._t("export_format.bmp_canvas"), "bmp_canvas"),
            (self._t("export_format.bmp_overlay_canvas"), "bmp_overlay_canvas"),
            (self._t("export_format.bmp"), "bmp"),
            (self._t("export_format.png"), "png"),
            (self._t("export_format.jpg"), "jpg"),
        )
        labels = [label for label, _value in options]
        selected, accepted = QInputDialog.getItem(
            self._view,
            self._t("dialog.export_grid_check_format_title"),
            self._t("dialog.export_grid_check_format_label"),
            labels,
            0,
            False,
        )
        if not accepted:
            return None
        selected_text = str(selected)
        return next((value for label, value in options if label == selected_text), "bmp_canvas")

    def _select_grid_check_canvas_size(
        self,
        state: ExtendMatrixTabState,
        records: tuple[FrameRecord, ...],
    ) -> tuple[int, int] | None:
        payloads = dict(getattr(state, "grid_inspection_payload_by_key", {}) or {})
        first_result = next(
            (
                payloads.get(str(getattr(record, "key", "") or ""))
                for record in records
                if payloads.get(str(getattr(record, "key", "") or "")) is not None
            ),
            None,
        )
        columns = max(1, int(getattr(state.layout_config, "frames_per_row", 1) or 1))
        rows = max(1, (len(records) + columns - 1) // columns)
        frame_width = max(1, int(getattr(first_result, "image_width", 0) or 1))
        frame_height = max(1, int(getattr(first_result, "image_height", 0) or 1))
        default_width = min(65535, frame_width * columns)
        default_height = min(65535, frame_height * rows)
        width, accepted = QInputDialog.getInt(
            self._view,
            self._t("dialog.export_grid_check_canvas_title"),
            self._t("dialog.export_grid_check_canvas_width"),
            default_width,
            1,
            65535,
            1,
        )
        if not accepted:
            return None
        height, accepted = QInputDialog.getInt(
            self._view,
            self._t("dialog.export_grid_check_canvas_title"),
            self._t("dialog.export_grid_check_canvas_height"),
            default_height,
            1,
            65535,
            1,
        )
        if not accepted:
            return None
        return int(width), int(height)

    @staticmethod
    def _grid_check_export_format_label(image_format: str) -> str:
        normalized = str(image_format or "bmp").strip().lower().lstrip(".")
        if normalized in {"jpg", "jpeg"}:
            return "JPG"
        if normalized == "png":
            return "PNG"
        return "BMP"

    def _export_grid_inspection_bmps(
        self,
        state: ExtendMatrixTabState,
        records: tuple[FrameRecord, ...],
        *,
        render_records: tuple[FrameRecord, ...] | None = None,
        image_format: str | None = None,
    ) -> None:
        selected_format = str(image_format or "")
        if not selected_format:
            selected_format = self._select_grid_check_export_format() or ""
        if not selected_format:
            return
        format_label = self._grid_check_export_format_label(selected_format)
        export_folder = self._ensure_export_folder()
        if export_folder is None:
            return
        payloads = dict(getattr(state, "grid_inspection_payload_by_key", {}) or {})
        if not payloads:
            QMessageBox.warning(
                self._view,
                self._t("dialog.warning_title"),
                self._t("message.no_grid_check_results"),
            )
            return
        target_records = tuple(record for record in records if record is not None)
        if not target_records:
            target_records = self._grid_inspection_records_with_results(state)
        canvas_size = None
        canvas_columns = max(1, int(getattr(state.layout_config, "frames_per_row", 1) or 1))
        if selected_format in {"bmp_canvas", "bmp_overlay_canvas"}:
            canvas_size = self._select_grid_check_canvas_size(state, target_records)
            if canvas_size is None:
                return
            placements, canvas_columns, _canvas_rows = build_matrix_layout(list(target_records), state.layout_config)
            target_records = tuple(record for record, _row, _column in placements)
        render_record_keys = None
        if render_records is not None:
            render_record_keys = tuple(
                str(getattr(record, "key", "") or "")
                for record in render_records
                if record is not None and str(getattr(record, "key", "") or "")
            )
        total_units = max(1, len(target_records))
        progress = QProgressDialog(
            self._t(
                "message.export_grid_check_bmps_progress", format=format_label, current=0, total=total_units, frame=""
            ),
            self._t("common.cancel"),
            0,
            total_units,
            self._view,
        )
        progress.setWindowTitle(self._t("context.export_grid_check_bmps"))
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        last_progress_update = 0.0

        def on_progress(current: int, total: int, frame_name: str) -> None:
            nonlocal last_progress_update
            now = perf_counter()
            if int(current) < int(total) and now - last_progress_update < 0.10:
                return
            last_progress_update = now
            progress.setMaximum(max(1, int(total)))
            progress.setValue(max(0, min(int(current), max(1, int(total)))))
            progress.setLabelText(
                self._t(
                    "message.export_grid_check_bmps_progress",
                    format=format_label,
                    current=int(current),
                    total=int(total),
                    frame=str(frame_name or ""),
                )
            )
            QApplication.processEvents()

        config_payload = dict(getattr(state, "grid_inspection_config_payload", {}) or {})
        enabled_error_types = tuple(
            str(item) for item in (config_payload.get("enabled_error_types") or ()) if str(item)
        )
        try:
            if selected_format in {"bmp_canvas", "bmp_overlay_canvas"} and canvas_size is not None:
                result = export_grid_cell_defect_canvas(
                    state.build_result,
                    {str(key): value for key, value in payloads.items()},
                    export_folder,
                    canvas_width=canvas_size[0],
                    canvas_height=canvas_size[1],
                    frames_per_row=canvas_columns,
                    records=target_records,
                    render_record_keys=render_record_keys,
                    enabled_reason_types=enabled_error_types or None,
                    overlay_errors_on_source_mask=selected_format == "bmp_overlay_canvas",
                    file_name="check_matrix_errors.bmp"
                    if selected_format == "bmp_overlay_canvas"
                    else "check_matrix.bmp",
                    progress_callback=on_progress,
                    cancel_check=progress.wasCanceled,
                )
            else:
                result = export_grid_cell_defect_bmps(
                    state.build_result,
                    {str(key): value for key, value in payloads.items()},
                    export_folder,
                    records=target_records,
                    render_record_keys=render_record_keys,
                    image_format=selected_format,
                    enabled_reason_types=enabled_error_types or None,
                    progress_callback=on_progress,
                    cancel_check=progress.wasCanceled,
                )
        except Exception as error:
            progress.close()
            QMessageBox.warning(self._view, self._t("dialog.warning_title"), str(error))
            return
        progress.close()
        if bool(result.get("cancelled", False)):
            return
        count = int(result.get("exported_count", 0))
        skipped = int(result.get("skipped_count", 0))
        destination = str(result.get("destination") or "")
        if count <= 0:
            errors = tuple(result.get("errors", ()) or ())
            message = (
                "\n".join(str(item) for item in errors[:10]) if errors else self._t("message.no_grid_check_results")
            )
            QMessageBox.warning(self._view, self._t("dialog.warning_title"), message)
            return
        if selected_format in {"bmp_canvas", "bmp_overlay_canvas"} and canvas_size is not None:
            message_key = (
                "message.export_grid_check_overlay_canvas_done"
                if selected_format == "bmp_overlay_canvas"
                else "message.export_grid_check_canvas_done"
            )
            message = self._t(
                message_key,
                width=canvas_size[0],
                height=canvas_size[1],
                skipped=skipped,
                file=destination,
            )
        else:
            message = self._t(
                "message.export_grid_check_bmps_done",
                format=format_label,
                count=count,
                skipped=skipped,
                folder=destination,
            )
        QMessageBox.information(self._view, self._t("dialog.info_title"), message)

    def _store_details_view_payload(self, payload: dict[str, object]) -> None:
        self._details_view_payload = dict(payload or {})
        self._settings_service.save_details_view_payload(self._details_view_payload)

    def _set_details_preferred_model_id(self, model_id: str | None) -> None:
        normalized = str(model_id or "") or None
        self._details_view_payload["preferred_model_id"] = normalized
        self._settings_service.save_details_view_payload(self._details_view_payload)
        for dialog in list(self._details_dialogs):
            setter = getattr(dialog, "set_preferred_model_id", None)
            if callable(setter):
                setter(normalized)

    def _preferred_details_model_id_for_state(
        self, state: ExtendMatrixTabState, *, session_view_state: dict[str, object] | None = None
    ) -> str | None:
        current_item = self.folder_list.currentItem() if hasattr(self, "folder_list") else None
        if current_item is not None:
            current_item_model_id = self._model_id_for_folder_item(current_item, state.build_result)
            if current_item_model_id:
                return current_item_model_id
        if state.metric_scope:
            return str(state.metric_scope)
        if state.confidence_model_id:
            return str(state.confidence_model_id)
        if session_view_state is not None:
            preferred_model_id = session_view_state.get("preferred_model_id")
            if isinstance(preferred_model_id, str) and preferred_model_id:
                return preferred_model_id
        return None

    def _default_details_result_kind_for_state(self, state: ExtendMatrixTabState) -> str:
        metric_key = str(
            getattr(state, "metric_key", "") or getattr(state.build_result, "selected_metric_key", "") or ""
        )
        if confidence_metric_family(metric_key) is not None:
            return "confidence"
        analysis_mode = str(getattr(state, "analysis_mode", "") or INTER_MODEL_ANALYSIS_MODE)
        if analysis_mode == CONFIDENCE_COMPARISON_MODE:
            return "diff"
        if analysis_mode in {INTRA_MODEL_CONFIDENCE_MODE, MODEL_OUTPUT_CONFIDENCE_MODE}:
            return "confidence"
        if str(getattr(state, "object_type", "") or POLYGON_OBJECT_TYPE) == POINT_OBJECT_TYPE:
            return "point_matches"
        return "diff"

    @staticmethod
    def _default_details_layer_view_for_state(_state: ExtendMatrixTabState) -> str:
        return "source"

    def _default_details_comparison_mode_for_state(self, state: ExtendMatrixTabState) -> str:
        comparison_mode = getattr(state.build_result.options, "comparison_mode", "")
        return str(getattr(comparison_mode, "value", comparison_mode) or "disagreement")

    def _default_details_grayscale_diff_for_state(self, state: ExtendMatrixTabState) -> bool:
        if self._default_details_result_kind_for_state(state) != "diff":
            return False
        if str(getattr(state, "analysis_mode", "") or INTER_MODEL_ANALYSIS_MODE) == CONFIDENCE_COMPARISON_MODE:
            return True
        comparison_mode = getattr(state.build_result.options, "comparison_mode", "")
        return str(getattr(comparison_mode, "value", comparison_mode) or "") == "grayscale_diff"

    def _update_matrix_preview(self, state: ExtendMatrixTabState, record: FrameRecord | None = None) -> None:
        selected = record or state.matrix_view.current_record()
        preview = state.preview
        if preview is None:
            return
        if selected is None:
            preview.frame_value.setText("-")
            for card in preview.score_cards.values():
                card.set_payload(
                    "-",
                    self._metric_score_style(None, state.metric_key),
                    "",
                    visible=False,
                    percentile_text="",
                    percentile_style=self._percentile_style(None),
                )
            if preview.overall_group is not None:
                preview.overall_group.hide()
            if preview.component_group is not None:
                preview.component_group.hide()
            return
        if self._record_is_excluded(state, selected):
            excluded_text = self._t("matrix.validation_na_excluded")
            preview.frame_value.setText(selected.display_name)
            if excluded_text:
                preview.frame_value.setText(f"{preview.frame_value.text()}\n{excluded_text}")
            for card in preview.score_cards.values():
                card.set_payload(
                    "-",
                    self._metric_score_style(None, state.metric_key),
                    "",
                    visible=False,
                    percentile_text="",
                    percentile_style=self._percentile_style(None),
                )
            if preview.overall_group is not None:
                preview.overall_group.hide()
            if preview.component_group is not None:
                preview.component_group.hide()
            return
        preview.frame_value.setText(selected.display_name)
        summary = selected.summary
        percentile_cache: dict[str, dict[str, float]] = {}
        visible_metric_keys = set(self._display_metric_keys_for_state(state, state.build_result))
        overall_visible = False
        component_visible = False
        for metric_key, card in preview.score_cards.items():
            value = metric_value_for_record(selected, metric_key) if summary is not None else None
            visible = metric_key in visible_metric_keys and value is not None
            details = (
                "\n".join(
                    self._decorate_metric_lines(
                        metric_key, summary, self._metric_component_lines(state, selected, metric_key)
                    )
                )
                if visible
                else ""
            )
            percentile_map = (
                percentile_cache.setdefault(metric_key, self._percentile_map_for_metric(state, metric_key))
                if visible
                else {}
            )
            percentile_value = percentile_map.get(selected.key) if visible else None
            tooltip = (
                self._metric_hint(metric_key, summary)
                if summary is not None
                else self._metric_hint_fallback(metric_key, state.build_result)
            )
            card.set_payload(
                self._metric_score_text(value, metric_key),
                self._metric_score_style(value, metric_key),
                details,
                visible=visible,
                percentile_text=self._percentile_text(percentile_value) if visible else "",
                percentile_style=self._percentile_style(percentile_value),
                tooltip=tooltip or "",
            )
            if visible:
                if str(metric_key).startswith("overall_"):
                    overall_visible = True
                else:
                    component_visible = True
        if preview.overall_group is not None:
            preview.overall_group.setVisible(overall_visible)
        if preview.component_group is not None:
            preview.component_group.setVisible(component_visible)

    def _sync_action_buttons(self) -> None:
        current_state = self._current_tab_state()
        active_model_count = len(self._checked_model_specs())
        can_build_from_base_only = (
            active_model_count <= 0 and self._original_folder is not None and Path(self._original_folder.path).exists()
        )
        is_busy = self._worker_thread is not None
        self._sync_confidence_map_function_state(
            None if current_state is None else current_state.build_result,
            allow_fallback=not is_busy,
        )
        required_model_count = self._required_model_count_for_build()
        self.btn_clear_folders.setEnabled(self.folder_list.count() > 0 and not is_busy)
        self.btn_set_original.setEnabled(not is_busy)
        self.btn_clear_original.setEnabled(self._original_folder is not None and not is_busy)
        self.btn_set_export.setEnabled(not is_busy)
        self.btn_clear_export.setEnabled(self._export_folder is not None and not is_busy)
        can_start_build = active_model_count >= required_model_count or can_build_from_base_only
        self.btn_build.setEnabled((current_state is not None or can_start_build) and not is_busy)
        self.btn_compute.setEnabled(current_state is not None and not is_busy)
        if hasattr(self, "frame_search_input"):
            self.frame_search_input.setEnabled(current_state is not None and not is_busy)
        if hasattr(self, "btn_frame_search"):
            self.btn_frame_search.setEnabled(current_state is not None and not is_busy)
        if hasattr(self, "btn_export_layer"):
            self.btn_export_layer.setEnabled(
                current_state is not None and bool(getattr(current_state.build_result, "records", ())) and not is_busy
            )
        if hasattr(self, "btn_export_grid_checks"):
            grid_mode = self._current_app_mode() == "grid_inspection"
            has_grid_payloads = (
                bool(getattr(current_state, "grid_inspection_payload_by_key", {}) or {})
                if current_state is not None
                else False
            )
            self.btn_export_grid_checks.setVisible(grid_mode)
            self.btn_export_grid_checks.setEnabled(grid_mode and has_grid_payloads and not is_busy)
        self.btn_cancel.setEnabled(is_busy)
        report = self._refresh_analysis_preflight()
        if hasattr(self._view, "set_analysis_profile_availability"):
            self._view.set_analysis_profile_availability(self._analysis_profile_availability())
        if hasattr(self._view, "set_analysis_profile"):
            self._view.set_analysis_profile(self._analysis_profile)
        if hasattr(self._view, "analysis_setup_panel"):
            self._view.analysis_setup_panel.set_preflight(report)
            self._view.analysis_setup_panel.set_busy(is_busy)
        if hasattr(self, "pair_matrix_table"):
            self.pair_matrix_table.setEnabled(not is_busy)
        if hasattr(self, "active_pair_list"):
            self.active_pair_list.setEnabled(not is_busy)
        if hasattr(self._view, "set_workflow_summary"):
            original_state = (
                self._t("workflow.state.ready")
                if self._original_folder is not None
                else self._t("workflow.state.pending")
            )
            sources_tone = "ready" if self._original_folder is not None else "warn"
            models_status = (
                self._t("workflow.state.ready") if active_model_count > 0 else self._t("workflow.state.pending")
            )
            models_tone = "ready" if active_model_count > 0 else "warn"
            if is_busy:
                analysis_status = self._t("workflow.state.running")
                analysis_detail = self.build_progress.format() or self._t("workflow.analysis_running")
                analysis_tone = "busy"
            elif current_state is None:
                analysis_status = self._t("workflow.state.pending")
                analysis_detail = self._t("workflow.analysis_pending")
                analysis_tone = "idle"
            elif bool(getattr(current_state.build_result, "scores_computed", False)):
                analysis_status = self._t("workflow.state.computed")
                analysis_detail = self._t("workflow.analysis_computed")
                analysis_tone = "active"
            else:
                analysis_status = self._t("workflow.state.built")
                analysis_detail = self._t("workflow.analysis_built")
                analysis_tone = "ready"
            self._view.set_workflow_summary(
                {
                    "sources": (
                        original_state,
                        self._t("workflow.sources_detail", original=original_state),
                        sources_tone,
                    ),
                    "models": (
                        models_status,
                        self._t("workflow.models_detail", count=active_model_count),
                        models_tone,
                    ),
                    "analysis": (analysis_status, analysis_detail, analysis_tone),
                }
            )

    def _build_folder_manager_payload(self) -> dict:
        return {
            "folders": [
                {
                    "path": str(self.folder_list.item(row).data(Qt.ItemDataRole.UserRole)),
                    "checked": bool(self.folder_list.item(row).data(FOLDER_CHECKED_ROLE)),
                    "label": str(self.folder_list.item(row).data(FOLDER_LABEL_ROLE) or ""),
                    "confidence_path": str(self.folder_list.item(row).data(FOLDER_CONFIDENCE_ROLE) or ""),
                    "confidence_expanded": bool(self.folder_list.item(row).data(FOLDER_CONFIDENCE_EXPANDED_ROLE)),
                }
                for row in range(self.folder_list.count())
            ],
            "original_folder": str(self._original_folder.path) if self._original_folder is not None else None,
            "export_folder": str(self._export_folder) if self._export_folder is not None else None,
            "comparison_pairs": [
                {
                    "model_a_id": pair.model_a_id,
                    "model_b_id": pair.model_b_id,
                    "operations": list(pair.operations),
                }
                for pair in self._selected_comparison_pairs()
            ],
            "comparison_pair_defaults_initialized": bool(self._pair_defaults_initialized),
        }

    def _restore_persisted_state(self) -> None:
        self._restore_folder_manager_state()
        self._restore_build_settings()
        self._update_source_labels()
        self._refresh_pair_matrix()
        self._on_app_mode_changed()

    def _restore_folder_manager_state(self) -> None:
        payload = self._settings_service.load_folder_manager_payload()
        if not payload:
            return
        self.folder_list.blockSignals(True)
        try:
            for folder_entry in payload.get("folders", []):
                path = folder_entry.get("path")
                if not path:
                    continue
                folder_path = Path(path)
                if not folder_path.exists():
                    continue
                item = self._append_folder_item(folder_path, checked=bool(folder_entry.get("checked", False)))
                item.setData(FOLDER_LABEL_ROLE, str(folder_entry.get("label") or folder_path.name))
                confidence_path = folder_entry.get("confidence_path")
                if confidence_path and Path(confidence_path).exists():
                    item.setData(FOLDER_CONFIDENCE_ROLE, str(confidence_path))
                item.setData(FOLDER_CONFIDENCE_EXPANDED_ROLE, bool(folder_entry.get("confidence_expanded", False)))
            original_folder = payload.get("original_folder")
            export_folder = payload.get("export_folder")
            if original_folder and Path(original_folder).exists():
                path = Path(original_folder)
                self._original_folder = FolderSpec(path=path, label=path.name)
            if export_folder and Path(export_folder).exists():
                self._export_folder = Path(export_folder)
            self._pair_defaults_initialized = bool(payload.get("comparison_pair_defaults_initialized", False))
            self._comparison_pair_operations.clear()
            for pair_entry in payload.get("comparison_pairs", []) or []:
                model_a = str(pair_entry.get("model_a_id") or "").strip()
                model_b = str(pair_entry.get("model_b_id") or "").strip()
                operations = {
                    str(operation).strip().lower()
                    for operation in pair_entry.get("operations", []) or []
                    if str(operation).strip().lower() in PAIR_OPERATION_ORDER
                }
                if model_a and model_b and operations:
                    self._comparison_pair_operations[(model_a, model_b)] = operations
        finally:
            self.folder_list.blockSignals(False)
        self._refresh_pair_matrix()

    def _build_build_settings_payload(self) -> dict:
        mask_threshold, boundary_radius = self._selected_polygon_compare_values()
        return {
            "thumbnail_size": int(DEFAULT_CELL_SIZE),
            "matrix_score_view_mode": str(self.matrix_score_view_combo.currentData() or DEFAULT_MATRIX_SCORE_VIEW_MODE),
            "gradient_name": str(self.matrix_gradient_combo.currentData() or DEFAULT_GRADIENT_NAME),
            "analysis_profile": self._analysis_profile.value,
            "analysis_mode": self._selected_analysis_mode(),
            "comparison_target": self._selected_comparison_target().value,
            "object_type": self._selected_object_type(),
            "geometry_mode": str(self.geometry_mode_combo.currentData() or DEFAULT_GEOMETRY_MODE),
            "polygon_compare_profile": self._selected_polygon_compare_profile(),
            "mask_threshold": float(mask_threshold),
            "boundary_radius": int(boundary_radius),
            "confidence_uncertainty_profile": self._selected_confidence_uncertainty_profile(),
            "confidence_uncertainty_delta": self._selected_confidence_uncertainty_delta(),
            "point_match_radius": float(self.point_match_radius_spin.value()),
            "point_confidence_radius": int(self.point_confidence_radius_spin.value()),
            "point_extraction_mode": str(
                self.point_extraction_mode_combo.currentData() or DEFAULT_POINT_EXTRACTION_MODE
            ),
            "polygon_confidence_summary": str(
                self.polygon_confidence_summary_combo.currentData() or DEFAULT_POLYGON_CONFIDENCE_SUMMARY
            ),
            "grid_inspection_config": self._grid_inspection_config_payload(),
            "layout_mode": "indexed_grid",
            "total_frames": int(DEFAULT_TOTAL_FRAMES),
            "frames_per_row": int(self.frames_per_row_spin.value()),
            "rows": int(DEFAULT_MATRIX_ROWS),
            "columns": int(DEFAULT_MATRIX_COLUMNS),
            "metric_key": str(self.metric_combo.currentData() or DEFAULT_MATRIX_METRIC_KEY),
            "metric_scope": str(self.metric_scope_combo.currentData() or ""),
            "confidence_model_id": str(self.metric_scope_combo.currentData() or ""),
            "frame_type_filter": str(self.frame_type_filter_combo.currentData() or "all"),
            "pair_panel_expanded": bool(self.pair_matrix_group.isChecked())
            if hasattr(self, "pair_matrix_group")
            else False,
            "analysis_panel_expanded": bool(self.analysis_settings_group.isChecked())
            if hasattr(self, "analysis_settings_group")
            else True,
        }

    def _build_analysis_profile_payload(self) -> dict[str, object]:
        bindings: list[AnalysisSourceBinding] = []
        if self._original_folder is not None:
            bindings.append(
                AnalysisSourceBinding(
                    binding_key="original",
                    role=AnalysisSourceRole.ORIGINAL,
                    kind=SourceBindingKind.FILESYSTEM,
                    source_id=str(self._original_folder.path),
                    display_name=self._original_folder.label,
                )
            )
        for spec in self._checked_model_specs():
            bindings.append(
                AnalysisSourceBinding(
                    binding_key=f"model:{spec.model_id}",
                    role=AnalysisSourceRole.MODEL_OUTPUT,
                    kind=SourceBindingKind.FILESYSTEM,
                    source_id=str(spec.mask_folder),
                    display_name=spec.display_name,
                )
            )
            if spec.prob_folder is not None:
                bindings.append(
                    AnalysisSourceBinding(
                        binding_key=f"confidence:{spec.model_id}",
                        role=AnalysisSourceRole.CONFIDENCE,
                        kind=SourceBindingKind.FILESYSTEM,
                        source_id=str(spec.prob_folder),
                        display_name=spec.display_name,
                    )
                )
        score_view = str(self.matrix_score_view_combo.currentData() or DEFAULT_MATRIX_SCORE_VIEW_MODE)
        profile = KarakalAnalysisProfileV1(
            profile=self._analysis_profile,
            bindings=tuple(bindings),
            object_type=self._selected_object_type(),
            metric_key=str(self.metric_combo.currentData() or DEFAULT_MATRIX_METRIC_KEY),
            scale_mode=AnalysisScaleMode.ABSOLUTE if score_view == "absolute" else AnalysisScaleMode.WITHIN_RUN,
            gradient_name=str(self.matrix_gradient_combo.currentData() or DEFAULT_GRADIENT_NAME),
            visible_layers=("quality", "status", "reference", "anomalies"),
            parameters=(
                AnalysisParameter("analysis_mode", self._selected_analysis_mode()),
                AnalysisParameter("comparison_target", self._selected_comparison_target().value),
                AnalysisParameter("frames_per_row", int(self.frames_per_row_spin.value())),
            ),
        )
        return profile.to_payload()

    def _restore_build_settings(self) -> None:
        payload = self._settings_service.load_build_settings_payload() or {}
        versioned_profile_payload = self._settings_service.load_analysis_profile_payload()
        if versioned_profile_payload:
            try:
                versioned_profile = KarakalAnalysisProfileV1.from_payload(versioned_profile_payload)
            except (TypeError, ValueError):
                versioned_profile = None
            if versioned_profile is not None:
                payload.setdefault("analysis_profile", versioned_profile.profile.value)
                payload.setdefault("gradient_name", versioned_profile.gradient_name)
                payload.setdefault("metric_key", versioned_profile.metric_key)
                payload.setdefault(
                    "matrix_score_view_mode",
                    "absolute" if versioned_profile.scale_mode == AnalysisScaleMode.ABSOLUTE else "relative",
                )
        blockers = [
            QSignalBlocker(self.thumbnail_size_spin),
            QSignalBlocker(self.matrix_score_view_combo),
            QSignalBlocker(self.matrix_gradient_combo),
            QSignalBlocker(self.analysis_mode_combo),
            QSignalBlocker(self.comparison_target_combo),
            QSignalBlocker(self.geometry_mode_combo),
            QSignalBlocker(self.polygon_compare_profile_combo),
            QSignalBlocker(self.mask_threshold_spin),
            QSignalBlocker(self.boundary_radius_spin),
            QSignalBlocker(self.confidence_uncertainty_profile_combo),
            QSignalBlocker(self.point_match_radius_spin),
            QSignalBlocker(self.point_confidence_radius_spin),
            QSignalBlocker(self.point_extraction_mode_combo),
            QSignalBlocker(self.polygon_confidence_summary_combo),
            QSignalBlocker(self.layout_mode_combo),
            QSignalBlocker(self.total_frames_spin),
            QSignalBlocker(self.frames_per_row_spin),
            QSignalBlocker(self.matrix_rows_spin),
            QSignalBlocker(self.matrix_columns_spin),
            QSignalBlocker(self.metric_group_combo),
            QSignalBlocker(self.metric_scope_combo),
            QSignalBlocker(self.metric_combo),
            QSignalBlocker(self.frame_type_filter_combo),
        ]
        _ = blockers
        self.thumbnail_size_spin.setValue(int(DEFAULT_CELL_SIZE))
        score_view_mode = str(payload.get("matrix_score_view_mode") or DEFAULT_MATRIX_SCORE_VIEW_MODE)
        score_view_index = self.matrix_score_view_combo.findData(score_view_mode)
        self.matrix_score_view_combo.setCurrentIndex(score_view_index if score_view_index >= 0 else 0)
        gradient_name = str(payload.get("gradient_name") or DEFAULT_GRADIENT_NAME)
        gradient_index = self.matrix_gradient_combo.findData(gradient_name)
        self.matrix_gradient_combo.setCurrentIndex(gradient_index if gradient_index >= 0 else 0)
        self._analysis_profile = analysis_profile_definition(
            str(payload.get("analysis_profile") or DEFAULT_ANALYSIS_PROFILE)
        ).key
        if hasattr(self._view, "set_analysis_profile"):
            self._view.set_analysis_profile(self._analysis_profile)
        analysis_mode = str(payload.get("analysis_mode") or self._selected_analysis_mode())
        analysis_index = self.analysis_mode_combo.findData(analysis_mode)
        self.analysis_mode_combo.setCurrentIndex(analysis_index if analysis_index >= 0 else 0)
        comparison_target = str(payload.get("comparison_target") or DEFAULT_COMPARISON_TARGET)
        comparison_target_index = self.comparison_target_combo.findData(comparison_target)
        self.comparison_target_combo.setCurrentIndex(comparison_target_index if comparison_target_index >= 0 else 0)
        geometry_mode = str(
            payload.get("geometry_mode")
            or geometry_mode_for_object_type(payload.get("object_type")).value
            or DEFAULT_GEOMETRY_MODE
        )
        geometry_index = self.geometry_mode_combo.findData(geometry_mode)
        self.geometry_mode_combo.setCurrentIndex(geometry_index if geometry_index >= 0 else 0)
        compare_profile = str(payload.get("polygon_compare_profile") or "")
        self.mask_threshold_spin.setValue(float(payload.get("mask_threshold", self.mask_threshold_spin.value())))
        self.boundary_radius_spin.setValue(int(payload.get("boundary_radius", self.boundary_radius_spin.value())))
        if not compare_profile:
            compare_profile = self._polygon_compare_profile_for_values(
                self.mask_threshold_spin.value(), self.boundary_radius_spin.value()
            )
        compare_index = self.polygon_compare_profile_combo.findData(compare_profile)
        self.polygon_compare_profile_combo.setCurrentIndex(compare_index if compare_index >= 0 else 0)
        uncertainty_profile = str(payload.get("confidence_uncertainty_profile") or "")
        if not uncertainty_profile:
            uncertainty_profile = self._confidence_uncertainty_profile_for_value(
                payload.get("confidence_uncertainty_delta")
            )
        uncertainty_index = self.confidence_uncertainty_profile_combo.findData(uncertainty_profile)
        self.confidence_uncertainty_profile_combo.setCurrentIndex(uncertainty_index if uncertainty_index >= 0 else 0)
        self.point_match_radius_spin.setValue(
            float(payload.get("point_match_radius", self.point_match_radius_spin.value()))
        )
        self.point_confidence_radius_spin.setValue(
            int(payload.get("point_confidence_radius", DEFAULT_POINT_CONFIDENCE_RADIUS))
        )
        point_extraction_mode = str(payload.get("point_extraction_mode") or DEFAULT_POINT_EXTRACTION_MODE)
        point_mode_index = self.point_extraction_mode_combo.findData(point_extraction_mode)
        self.point_extraction_mode_combo.setCurrentIndex(point_mode_index if point_mode_index >= 0 else 0)
        polygon_confidence_summary = str(
            payload.get("polygon_confidence_summary") or DEFAULT_POLYGON_CONFIDENCE_SUMMARY
        )
        polygon_summary_index = self.polygon_confidence_summary_combo.findData(polygon_confidence_summary)
        self.polygon_confidence_summary_combo.setCurrentIndex(
            polygon_summary_index if polygon_summary_index >= 0 else 0
        )
        self._set_grid_inspection_config_controls(dict(payload.get("grid_inspection_config") or {}))
        layout_index = self.layout_mode_combo.findData("indexed_grid")
        self.layout_mode_combo.setCurrentIndex(layout_index if layout_index >= 0 else 0)
        self.total_frames_spin.setValue(int(DEFAULT_TOTAL_FRAMES))
        self.frames_per_row_spin.setValue(int(payload.get("frames_per_row", DEFAULT_FRAMES_PER_ROW)))
        self.matrix_rows_spin.setValue(int(DEFAULT_MATRIX_ROWS))
        self.matrix_columns_spin.setValue(int(DEFAULT_MATRIX_COLUMNS))
        metric_key = str(payload.get("metric_key") or DEFAULT_MATRIX_METRIC_KEY)
        metric_scope = str(
            payload.get("confidence_model_id")
            or payload.get("metric_scope")
            or self._metric_scope_for_metric_key(metric_key)
            or ""
        )
        frame_type_filter = str(payload.get("frame_type_filter") or self._selected_object_type())
        self._sync_metric_controls(None, preferred_metric_key=metric_key, preferred_scope_key=metric_scope)
        index = self.frame_type_filter_combo.findData(frame_type_filter)
        self.frame_type_filter_combo.setCurrentIndex(index if index >= 0 else 0)
        pair_panel_expanded = bool(payload.get("pair_panel_expanded", False))
        if hasattr(self, "pair_matrix_group"):
            blocker = QSignalBlocker(self.pair_matrix_group)
            self.pair_matrix_group.setChecked(pair_panel_expanded)
            del blocker
        if hasattr(self, "pair_matrix_body"):
            self.pair_matrix_body.setVisible(pair_panel_expanded)
        analysis_panel_expanded = bool(payload.get("analysis_panel_expanded", False))
        if hasattr(self, "analysis_settings_group"):
            blocker = QSignalBlocker(self.analysis_settings_group)
            self.analysis_settings_group.setChecked(analysis_panel_expanded)
            del blocker
        if hasattr(self, "analysis_settings_body"):
            self.analysis_settings_body.setVisible(analysis_panel_expanded)
        self._sync_mode_controls(None, None)

    def _persist_state(self) -> None:
        self._settings_service.save_folder_manager_payload(self._build_folder_manager_payload())
        self._settings_service.save_build_settings_payload(self._build_build_settings_payload())
        self._settings_service.save_analysis_profile_payload(self._build_analysis_profile_payload())
        self._settings_service.sync()

    def shutdown(self) -> None:
        if self._worker is not None:
            request_cancel = getattr(self._worker, "request_cancel", None)
            if callable(request_cancel):
                request_cancel()
        thread = self._worker_thread
        if thread is not None:
            thread.quit()
            if thread.isRunning():
                thread.wait(30000)
        self._cleanup_worker()
        self._close_all_details_dialogs()
        self._persist_state()

    # Preferred analytics entrypoint.
    def _start_compute_metrics(self) -> None:
        self._start_compute_analytics()

    # Legacy lite compatibility alias.
    def _start_compute_mismatches(self) -> None:
        self._start_compute_metrics()

    def _set_base_folder(self) -> None:
        self._set_original_folder()

    def _clear_base_folder(self) -> None:
        self._clear_original_folder()
