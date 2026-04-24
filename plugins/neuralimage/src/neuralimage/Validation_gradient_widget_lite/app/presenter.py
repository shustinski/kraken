"""Presenter for the extended validation gradient widget."""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import replace
from datetime import datetime
from math import isfinite
from pathlib import Path

from PyQt6.QtCore import QObject, QSignalBlocker, QThread, QTimer, Qt
from PyQt6.QtGui import QImageReader
from PyQt6.QtWidgets import QFileDialog, QListWidgetItem, QMessageBox

from ..core.analysis_modes import (
    INTRA_MODEL_CONFIDENCE_MODE,
    INTER_MODEL_ANALYSIS_MODE,
    POINT_OBJECT_TYPE,
    POLYGON_OBJECT_TYPE,
    confidence_metric_key,
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
from ..core.backend_constants import (
    BCE_SCORE_CAP,
)
from ..core.domain import BuildOptions, BuildResult, FolderSpec, FrameIdentity, FrameRecord, GeometryMode, ModelSpec
from ..core.repository import (
    _parse_model_metric_key,
    compute_metric_percentiles,
    load_grayscale_image,
    metric_higher_is_better,
    metric_value_for_record,
)
from ..core.subpixel_grid import SubpixelGridSpec, subpixel_spec_from_options
from ..core.tile_grid import TileGridPlan, plan_tile_grid
from ..core.workers import AnalyticsWorker, FrameIndexWorker
from ..infra.services import ValidationGradientLiteSettingsService
from ..ui.details_dialog import ExtendFrameDetailsDialog
from ..ui.matrix_view import MatrixLayoutConfig, build_matrix_layout
from ..ui.ui_components import FolderRowWidget
from ..ui.ui_constants import (
    DEFAULT_CELL_SIZE,
    DEFAULT_BOUNDARY_RADIUS,
    DEFAULT_CONFIDENCE_UNCERTAINTY_DELTA,
    DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE,
    DEFAULT_FRAMES_PER_ROW,
    DEFAULT_GEOMETRY_MODE,
    DEFAULT_GRADIENT_NAME,
    DEFAULT_MATRIX_COLUMNS,
    DEFAULT_MATRIX_LAYOUT_MODE,
    DEFAULT_MATRIX_SCORE_VIEW_MODE,
    DEFAULT_MATRIX_METRIC_KEY,
    DEFAULT_METRIC_SCOPE,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_MATRIX_ROWS,
    DEFAULT_SUBPIXEL_AGGREGATION,
    DEFAULT_SUBPIXEL_COLUMNS,
    DEFAULT_SUBPIXEL_ROWS,
    DEFAULT_SUBPIXEL_VIEW_MODE,
    DEFAULT_POINT_CONFIDENCE_RADIUS,
    DEFAULT_POINT_EXTRACTION_MODE,
    DEFAULT_POLYGON_CONFIDENCE_SUMMARY,
    DEFAULT_POLYGON_COMPARE_PROFILE,
    DEFAULT_TILE_HEIGHT,
    DEFAULT_TILE_MODE,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_OVERLAP_MODE,
    DEFAULT_TILE_WIDTH,
    DEFAULT_TOTAL_FRAMES,
    FOLDER_CHECKED_ROLE,
    FOLDER_CONFIDENCE_ROLE,
    FOLDER_LABEL_ROLE,
    FOLDER_ROW_MIN_HEIGHT,
    MATRIX_METRIC_GROUP_OPTIONS,
    MATRIX_METRIC_OPTIONS,
    PERCENTILE_BAND_BOUNDS,
    CONFIDENCE_UNCERTAINTY_PROFILE_VALUES,
    POLYGON_COMPARE_PROFILE_VALUES,
)
from .state import ExtendMatrixTabState


class ValidationGradientExtendPresenter(QObject):
    """Coordinate UI state, background workers and matrix tabs."""

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

    def _selected_polygon_compare_values(self) -> tuple[float, int]:
        values = self._polygon_compare_values_for_profile(self._selected_polygon_compare_profile())
        if values is None:
            values = self._polygon_compare_values_for_profile(DEFAULT_POLYGON_COMPARE_PROFILE)
        if values is None:
            return float(DEFAULT_MASK_THRESHOLD), int(DEFAULT_BOUNDARY_RADIUS)
        return values

    def _selected_tile_mode(self) -> str:
        return "tile" if self._selected_subpixel_view_mode() == "tile" else "pixel"

    def _selected_subpixel_view_mode(self) -> str:
        value = str(self.subpixel_view_mode_combo.currentData() or DEFAULT_SUBPIXEL_VIEW_MODE)
        if value == "tile":
            return "tile"
        return "pixel"

    def _selected_subpixel_rows(self) -> int:
        return int(self.subpixel_rows_spin.value())

    def _selected_subpixel_columns(self) -> int:
        return int(self.subpixel_columns_spin.value())

    def _selected_subpixel_aggregation(self) -> str:
        return str(self.subpixel_aggregation_combo.currentData() or DEFAULT_SUBPIXEL_AGGREGATION)

    def _selected_tile_width(self) -> int:
        return int(self.tile_width_spin.value())

    def _selected_tile_height(self) -> int:
        return int(self.tile_height_spin.value())

    def _selected_tile_overlap_mode(self) -> str:
        return str(self.tile_overlap_mode_combo.currentData() or DEFAULT_TILE_OVERLAP_MODE)

    def _selected_tile_overlap(self) -> int:
        return int(self.tile_overlap_spin.value())

    def _sync_tile_overlap_bounds(self) -> None:
        maximum = max(0, min(self._selected_tile_width(), self._selected_tile_height()) - 1)
        self.tile_overlap_spin.setMaximum(int(maximum))
        if int(self.tile_overlap_spin.value()) > maximum:
            self.tile_overlap_spin.setValue(int(maximum))

    @staticmethod
    def _set_row_enabled(row: object | None, enabled: bool) -> None:
        if row is not None and hasattr(row, "setEnabled"):
            row.setEnabled(bool(enabled))

    @staticmethod
    def _image_shape(path: Path) -> tuple[int, int] | None:
        try:
            reader = QImageReader(str(path))
            size = reader.size()
            if size.isValid() and size.width() > 0 and size.height() > 0:
                return int(size.height()), int(size.width())
        except Exception:
            pass
        try:
            return tuple(int(v) for v in load_grayscale_image(path).shape)
        except Exception:
            return None

    @staticmethod
    def _first_image_shape(folder_path: Path) -> tuple[int, int] | None:
        allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        try:
            if folder_path.is_file() and folder_path.suffix.lower() in allowed:
                return ValidationGradientExtendPresenter._image_shape(folder_path)
            for image_path in Path(folder_path).rglob("*"):
                if not image_path.is_file() or image_path.suffix.lower() not in allowed:
                    continue
                shape = ValidationGradientExtendPresenter._image_shape(image_path)
                if shape is not None:
                    return shape
        except Exception:
            return None
        return None

    def _tile_source_shape_for_result(self, build_result: BuildResult | None) -> tuple[int, int] | None:
        if build_result is None or not build_result.records:
            return None
        for record in build_result.records:
            candidate_path = record.original_path or record.base_path
            if not candidate_path and record.model_mask_paths:
                candidate_path = next(iter(record.model_mask_paths.values()), None)
            if not candidate_path:
                continue
            shape = self._image_shape(Path(candidate_path))
            if shape is not None:
                return shape
        original_folder = getattr(build_result, "original_folder", None)
        if original_folder is not None:
            shape = self._first_image_shape(Path(original_folder.path))
            if shape is not None:
                return shape
        return None

    def _tile_grid_plan_for_result(self, build_result: BuildResult | None) -> TileGridPlan | None:
        if self._selected_tile_mode() != "tile":
            return None
        source_shape = self._tile_source_shape_for_result(build_result)
        if source_shape is None:
            return None
        return plan_tile_grid(
            source_shape,
            self._selected_tile_width(),
            self._selected_tile_height(),
            self._selected_tile_overlap(),
        )

    def _tile_grid_layout_config_for_result(self, build_result: BuildResult | None) -> MatrixLayoutConfig | None:
        plan = self._tile_grid_plan_for_result(build_result)
        if plan is None or not plan.applied_exact:
            return None
        return MatrixLayoutConfig(
            mode="manual_grid",
            total_frames=max(1, int(plan.rows * plan.columns)),
            frames_per_row=max(1, int(plan.columns)),
            rows=max(1, int(plan.rows)),
            columns=max(1, int(plan.columns)),
        )

    def _subpixel_grid_spec(self, build_result: BuildResult | None = None) -> SubpixelGridSpec | None:
        if self._selected_subpixel_view_mode() != "tile":
            return None
        source_shape = self._tile_source_shape_for_result(build_result)
        options = BuildOptions(
            subpixel_view_mode="tile",
            subpixel_rows=max(1, self._selected_subpixel_rows()),
            subpixel_columns=max(1, self._selected_subpixel_columns()),
            tile_width=self._selected_tile_width(),
            tile_height=self._selected_tile_height(),
            tile_overlap=self._selected_tile_overlap(),
        )
        return subpixel_spec_from_options(options, source_shape)

    def _update_subpixel_plan_label(self, build_result: BuildResult | None) -> None:
        if self._selected_subpixel_view_mode() != "tile":
            self.tile_plan_label.setText(self._t("matrix.subpixel_plan.unavailable"))
            self.tile_plan_label.setToolTip("")
            self.tile_plan_label.setStyleSheet("")
            return
        spec = self._subpixel_grid_spec(build_result)
        if spec is None:
            self.tile_plan_label.setText(self._t("matrix.subpixel_plan.waiting"))
            self.tile_plan_label.setToolTip("")
            self.tile_plan_label.setStyleSheet("")
            return
        aggregation_label = self._t(f"subpixel_aggregation.{self._selected_subpixel_aggregation()}")
        self.tile_plan_label.setText(
            f"{int(spec.rows)}x{int(spec.columns)} | "
            f"{int(spec.tile_width)}x{int(spec.tile_height)} | "
            f"ovl {int(spec.overlap)} | {aggregation_label}"
        )
        self.tile_plan_label.setToolTip(self._t("matrix.subpixel_plan.tooltip"))
        self.tile_plan_label.setStyleSheet("")

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
        for row in range(self.folder_list.count()):
            existing_item = self.folder_list.item(row)
            if Path(existing_item.data(Qt.ItemDataRole.UserRole)) == folder_path:
                existing_item.setData(FOLDER_CHECKED_ROLE, bool(checked))
                if not existing_item.data(FOLDER_LABEL_ROLE):
                    existing_item.setData(FOLDER_LABEL_ROLE, folder_path.name)
                return existing_item
        item = QListWidgetItem()
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        item.setData(Qt.ItemDataRole.UserRole, str(folder_path))
        item.setData(FOLDER_CHECKED_ROLE, bool(checked))
        item.setData(FOLDER_LABEL_ROLE, folder_path.name)
        item.setData(FOLDER_CONFIDENCE_ROLE, "")
        item.setToolTip(str(folder_path))
        self.folder_list.addItem(item)
        return item

    @staticmethod
    def _folder_has_supported_images(folder_path: Path) -> bool:
        if not folder_path.exists() or not folder_path.is_dir():
            return False
        allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        try:
            for image_path in folder_path.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in allowed:
                    return True
        except Exception:
            return False
        return False

    def _refresh_folder_rows(self) -> None:
        for row in range(self.folder_list.count()):
            item = self.folder_list.item(row)
            path_text = str(item.data(Qt.ItemDataRole.UserRole))
            display_text = str(item.data(FOLDER_LABEL_ROLE) or (Path(path_text).name or path_text))
            confidence_path_text = str(item.data(FOLDER_CONFIDENCE_ROLE) or "")
            confidence_display_text = self._compact_path_text(confidence_path_text)
            row_widget = FolderRowWidget(
                self.folder_list,
                path_text=path_text,
                display_text=display_text,
                checked=bool(item.data(FOLDER_CHECKED_ROLE)),
                confidence_display_text=confidence_display_text,
                confidence_path_text=confidence_path_text,
                can_move_up=row > 0,
                can_move_down=row < self.folder_list.count() - 1,
                on_checked_changed=lambda checked, item=item: self._set_folder_item_checked(item, checked),
                on_label_changed=lambda text, item=item: self._set_folder_item_label(item, text),
                on_confidence_folder=lambda _checked=False, item=item: self._set_folder_item_confidence_folder(item),
                on_clear_confidence_folder=lambda _checked=False, item=item: self._clear_folder_item_confidence_folder(item),
                on_remove=lambda _checked=False, item=item: self._remove_folder_item(item),
                on_move_up=lambda _checked=False, item=item: self._move_folder_item(item, -1),
                on_move_down=lambda _checked=False, item=item: self._move_folder_item(item, 1),
                checkbox_tooltip="Use model in analytics",
                confidence_placeholder=self._t("folders.confidence_not_set"),
                confidence_tooltip=confidence_path_text,
                confidence_select_tooltip=self._t("folders.select_confidence"),
                confidence_clear_tooltip=self._t("folders.clear_confidence"),
                remove_tooltip="Remove model folder",
                move_up_tooltip="Move up",
                move_down_tooltip="Move down",
            )
            row_widget.setMinimumHeight(FOLDER_ROW_MIN_HEIGHT)
            item.setSizeHint(row_widget.sizeHint())
            self.folder_list.setItemWidget(item, row_widget)

    def _set_folder_item_checked(self, item: QListWidgetItem, checked: bool) -> None:
        self._folder_check_guard = True
        item.setData(FOLDER_CHECKED_ROLE, bool(checked))
        self._folder_check_guard = False
        self._refresh_folder_rows()
        self._sync_action_buttons()

    def _set_folder_item_label(self, item: QListWidgetItem, text: str) -> None:
        folder_path = Path(item.data(Qt.ItemDataRole.UserRole))
        item.setData(FOLDER_LABEL_ROLE, text or folder_path.name)
        self._refresh_folder_rows()

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

    def _remove_folder_item(self, item: QListWidgetItem) -> None:
        row = self.folder_list.row(item)
        if row < 0:
            return
        self.folder_list.takeItem(row)
        self._refresh_folder_rows()
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

    def _build_layout_config(self) -> MatrixLayoutConfig:
        return MatrixLayoutConfig(
            mode=str(self.layout_mode_combo.currentData() or DEFAULT_MATRIX_LAYOUT_MODE),
            total_frames=int(self.total_frames_spin.value()),
            frames_per_row=int(self.frames_per_row_spin.value()),
            rows=int(self.matrix_rows_spin.value()),
            columns=int(self.matrix_columns_spin.value()),
        )

    def _percentile_bin_bounds(self, bin_index: int) -> tuple[float, float]:
        normalized = max(0, min(int(bin_index), len(PERCENTILE_BAND_BOUNDS) - 1))
        low_bound, high_bound = PERCENTILE_BAND_BOUNDS[normalized]
        return float(low_bound), float(high_bound)

    def _records_in_percentile_bin(self, records: tuple[FrameRecord, ...] | list[FrameRecord], percentile_map: dict[str, float], bin_index: int) -> tuple[FrameRecord, ...]:
        low_bound, high_bound = self._percentile_bin_bounds(bin_index)
        selected: list[FrameRecord] = []
        for record in records:
            percentile = float(percentile_map.get(record.key, 0.0))
            if bin_index >= len(PERCENTILE_BAND_BOUNDS) - 1:
                matches = low_bound <= percentile <= high_bound
            else:
                matches = low_bound <= percentile < high_bound
            if matches:
                selected.append(record)
        return tuple(selected)

    def _base_records_for_state(self, state: ExtendMatrixTabState) -> tuple[FrameRecord, ...]:
        cache_key = (str(getattr(state, 'object_type', POLYGON_OBJECT_TYPE) or POLYGON_OBJECT_TYPE), id(state.build_result.records))
        cached = state.base_records_cache.get(cache_key)
        if cached is not None:
            return tuple(cached)
        records: tuple[FrameRecord, ...] | list[FrameRecord] = state.build_result.records
        object_type = str(getattr(state, 'object_type', POLYGON_OBJECT_TYPE) or POLYGON_OBJECT_TYPE)
        if object_type in {POLYGON_OBJECT_TYPE, POINT_OBJECT_TYPE}:
            records = tuple(
                record for record in records
                if record.summary is None or str(getattr(record.summary, 'frame_type', POLYGON_OBJECT_TYPE)) == object_type
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
        if state.percentile_filter_metric_key is not None and state.percentile_filter_bin_index is not None:
            percentile_map = self._percentile_map_for_metric(state, state.percentile_filter_metric_key)
            records = self._records_in_percentile_bin(records, percentile_map, state.percentile_filter_bin_index)
        if state.correlation_filter_band in {'bad', 'good'}:
            allowed_keys = {record.key for record, _count, _avg, _labels in self._repeated_percentile_entries(state, band=str(state.correlation_filter_band))[:25]}
            records = tuple(record for record in records if record.key in allowed_keys)
        return tuple(records)

    def _capture_view_snapshot(self) -> dict[str, object]:
        confidence_model_id = self._selected_confidence_model_id(None)
        mask_threshold, boundary_radius = self._selected_polygon_compare_values()
        return {
            "cell_size": int(DEFAULT_CELL_SIZE),
            "layout_config": self._build_layout_config(),
            "matrix_score_view_mode": str(self.matrix_score_view_combo.currentData() or DEFAULT_MATRIX_SCORE_VIEW_MODE),
            "analysis_mode": self._selected_analysis_mode(),
            "object_type": self._selected_object_type(),
            "geometry_mode": str(self.geometry_mode_combo.currentData() or DEFAULT_GEOMETRY_MODE),
            "polygon_compare_profile": self._selected_polygon_compare_profile(),
            "mask_threshold": float(mask_threshold),
            "boundary_radius": int(boundary_radius),
            "confidence_uncertainty_profile": self._selected_confidence_uncertainty_profile(),
            "confidence_uncertainty_delta": self._selected_confidence_uncertainty_delta(),
            "point_match_radius": float(self.point_match_radius_spin.value()),
            "point_confidence_radius": int(self.point_confidence_radius_spin.value()),
            "point_extraction_mode": str(self.point_extraction_mode_combo.currentData() or DEFAULT_POINT_EXTRACTION_MODE),
            "polygon_confidence_summary": str(self.polygon_confidence_summary_combo.currentData() or DEFAULT_POLYGON_CONFIDENCE_SUMMARY),
            "tile_mode": self._selected_tile_mode(),
            "tile_width": int(self.tile_width_spin.value()),
            "tile_height": int(self.tile_height_spin.value()),
            "tile_overlap_mode": self._selected_tile_overlap_mode(),
            "tile_overlap": int(self.tile_overlap_spin.value()),
            "subpixel_view_mode": self._selected_subpixel_view_mode(),
            "subpixel_rows": self._selected_subpixel_rows(),
            "subpixel_columns": self._selected_subpixel_columns(),
            "subpixel_aggregation": self._selected_subpixel_aggregation(),
            "metric_key": str(self.metric_combo.currentData() or DEFAULT_MATRIX_METRIC_KEY),
            "metric_scope": str(confidence_model_id or ""),
            "confidence_model_id": confidence_model_id,
            "frame_type_filter": str(self.frame_type_filter_combo.currentData() or 'all'),
        }

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
        elif family == "model_confidence":
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
        if str(metric_key) in {"overall_polygon_score", "iou_score", "dice_score", "polygon_bce_score", "overall_point_score", "precision_score", "recall_score", "f1_score", "localization_score"}:
            return f"{level} {float(value):.1f}"
        return f"{level} {float(value):.4f}"

    def _metric_label(self, metric_key: str, build_result: BuildResult | None = None) -> str:
        metric_key_text = str(metric_key)
        for label_key, key, _group in MATRIX_METRIC_OPTIONS:
            if str(key) == metric_key_text:
                return self._t(str(label_key))
        translated = self._t(f"metric.{metric_key_text}")
        if translated != f"metric.{metric_key_text}":
            return translated
        if '::' in metric_key_text:
            family, model_id = metric_key_text.split('::', 1)
            model_name = model_id
            if build_result is not None:
                for spec in build_result.model_specs:
                    if spec.model_id == model_id:
                        model_name = spec.display_name
                        break
            if family == 'model_confidence':
                return f"{self._t('metric.model_confidence')} [{model_name}]"
            if family == 'model_uncertain_fraction':
                return f"{self._t('metric.model_uncertain_fraction')} [{model_name}]"
            if family == 'model_point_contrast':
                return f"{self._t('metric.model_point_contrast')} [{model_name}]"
        return metric_key_text

    def _metric_hint(self, metric_key: str, summary) -> str | None:
        metric_key_text = str(metric_key)
        if '::' in metric_key_text:
            family, _model_id = metric_key_text.split('::', 1)
            if family in {'model_confidence', 'model_uncertain_fraction', 'model_point_contrast'}:
                if summary.frame_type == 'point':
                    return self._t('hint.intra_model_point')
                return self._t('hint.confidence_polygon')
        if metric_key_text == 'overall_frame_score':
            return self._t('hint.overall_labeled') if summary.is_labeled else self._t('hint.overall_unlabeled')
        if metric_key_text in {'overall_polygon_score', 'iou_score', 'dice_score', 'polygon_bce_score', 'iou', 'dice', 'bce'}:
            return self._t('hint.inter_model_polygon')
        if metric_key_text in {'overall_point_score', 'precision_score', 'recall_score', 'f1_score', 'localization_score', 'precision', 'recall', 'f1', 'mean_localization_distance'}:
            return self._t('hint.inter_model_point')
        if metric_key_text == 'model_model_score':
            return self._t('hint.model_model_point') if summary.frame_type == 'point' else self._t('hint.model_model_polygon')
        if metric_key_text in {'model_labeled_score', 'labeled_best_quality', 'labeled_mean_quality'}:
            return self._t('hint.model_labeled_point') if summary.frame_type == 'point' else self._t('hint.model_labeled_polygon')
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
        family = metric_key_text.split('::', 1)[0]
        defaults = {
            'overall_frame_score': self._t('hint.overall_unlabeled'),
            'model_model_score': self._t('hint.model_model_polygon'),
            'model_labeled_score': self._t('hint.model_labeled_polygon'),
            'labeled_best_quality': self._t('hint.model_labeled_polygon'),
            'labeled_mean_quality': self._t('hint.model_labeled_polygon'),
            'model_confidence': self._t('hint.confidence_polygon'),
            'model_uncertain_fraction': self._t('hint.confidence_polygon'),
            'model_point_contrast': self._t('hint.confidence_point'),
            'overall_polygon_score': self._t('hint.inter_model_polygon'),
            'iou_score': self._t('hint.inter_model_polygon'),
            'dice_score': self._t('hint.inter_model_polygon'),
            'polygon_bce_score': self._t('hint.inter_model_polygon'),
            'iou': self._t('hint.inter_model_polygon'),
            'dice': self._t('hint.inter_model_polygon'),
            'bce': self._t('hint.inter_model_polygon'),
            'overall_point_score': self._t('hint.inter_model_point'),
            'precision_score': self._t('hint.inter_model_point'),
            'recall_score': self._t('hint.inter_model_point'),
            'f1_score': self._t('hint.inter_model_point'),
            'localization_score': self._t('hint.inter_model_point'),
            'precision': self._t('hint.inter_model_point'),
            'recall': self._t('hint.inter_model_point'),
            'f1': self._t('hint.inter_model_point'),
            'mean_localization_distance': self._t('hint.inter_model_point'),
        }
        return defaults.get(family, self._metric_label(metric_key_text, build_result))

    def _metric_component_summary(self, metric_key: str, summary) -> str:
        metric_key_text = str(metric_key)
        family = metric_key_text.split('::', 1)[0]
        is_ru = getattr(self._i18n, 'language', 'en') == 'ru'
        if family == 'overall_frame_score':
            return self._t('metric.model_labeled_score') if summary.is_labeled else self._t('metric.disagreement_score')
        if family in {'overall_polygon_score', 'iou_score', 'dice_score', 'polygon_bce_score', 'iou', 'dice', 'bce'}:
            return 'iou + dice + bce score'
        if family in {'overall_point_score', 'precision_score', 'recall_score', 'f1_score', 'localization_score', 'precision', 'recall', 'f1', 'mean_localization_distance'}:
            return 'precision + recall + f1 + localization + tp/fp/fn'
        if family == 'model_model_score':
            return 'soft_dice + soft_iou + ssim + dice + iou + hausdorff + centroid' if summary.frame_type != 'point' else ('precision + recall + f1_at_r + локализация + количество' if is_ru else 'precision + recall + f1_at_r + localization + count')
        if family in {'model_labeled_score', 'labeled_best_quality', 'labeled_mean_quality'}:
            return 'soft_dice + soft_iou + ssim + dice + iou + hausdorff + centroid' if summary.frame_type != 'point' else ('precision + recall + f1_at_r + локализация + количество' if is_ru else 'precision + recall + f1_at_r + localization + count')
        if family == 'model_confidence':
            return 'средняя уверенность внутри объекта' if (is_ru and summary.frame_type != 'point') else ('средняя уверенность по точкам' if is_ru else ('mean object confidence' if summary.frame_type != 'point' else 'mean point confidence'))
        if family == 'model_uncertain_fraction':
            return 'доля сомнительных пикселей объекта' if is_ru else 'uncertain object fraction'
        if family == 'model_point_contrast':
            return 'средний контраст точек' if is_ru else 'mean point contrast'
        if family == 'disagreement_score':
            return '1 - согласие моделей' if is_ru else '1 - model-to-model agreement'
        return '-'

    def _component_name_label(self, name: str) -> str:
        labels_en = {
            'source': 'Source',
            'formula': 'Formula',
            'definition': 'Definition',
            'value': 'Value',
            'model': 'Model',
            'labeled_best_quality': 'Best labeled quality',
            'frame_uncertainty_score': 'Frame uncertainty score',
            'mean_uncertainty': 'Mean uncertainty',
            'low_conf_fraction': 'Low-confidence fraction',
            'worst_tail_uncertainty': 'Worst-tail uncertainty',
            'largest_low_conf_component': 'Largest low-confidence component',
            'uncertain_support_fraction': 'Uncertain support fraction',
            'top_uncertainty_mean': 'Top uncertainty mean',
            'largest_uncertain_region_fraction': 'Largest uncertain region fraction',
            'mean_object_confidence': 'Mean object confidence',
            'mean_object_probability': 'Mean object probability',
            'uncertain_fraction': 'Uncertain fraction',
            'object_area_fraction': 'Object area fraction',
            'mean_point_confidence': 'Mean point confidence',
            'mean_point_probability': 'Mean point probability',
            'mean_point_contrast': 'Mean point contrast',
            'point_count': 'Point count',
            'soft_dice': 'Soft Dice',
            'soft_iou': 'Soft IoU',
            'ssim': 'SSIM',
            'dice': 'Dice',
            'iou': 'IoU',
            'hausdorff_distance': 'Hausdorff distance',
            'centroid_distance': 'Centroid distance',
            'mae': 'MAE',
            'rmse': 'RMSE',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1',
            'f1_at_r': 'F1@r',
            'tp': 'TP',
            'fp': 'FP',
            'fn': 'FN',
            'bce': 'BCE',
            'iou_score': 'IoU score',
            'dice_score': 'Dice score',
            'polygon_bce_score': 'BCE score',
            'overall_polygon_score': 'Overall polygon score',
            'precision_score': 'Precision score',
            'recall_score': 'Recall score',
            'f1_score': 'F1 score',
            'overall_point_score': 'Overall point score',
            'mean_localization_distance': 'Mean localization distance',
            'mean_localization_error': 'Mean localization error',
            'localization_score': 'Localization score',
            'localization_agreement': 'Localization agreement',
            'count_error': 'Count error',
            'count_agreement': 'Count agreement',
            'connected_component_error': 'Connected-component error',
            'cc_error': 'Connected-component error',
            'chamfer_score': 'Chamfer score',
            'hausdorff_score': 'Hausdorff score',
        }
        labels_ru = {
            'source': 'Источник',
            'formula': 'Формула',
            'definition': 'Определение',
            'value': 'Значение',
            'model': 'Модель',
            'hot_region_count': 'Число горячих областей',
            'labeled_best_quality': 'Лучшее качество на размеченных кадрах',
            'acquisition_score': 'Приоритет на разметку',
            'mean_object_confidence': 'Средняя уверенность внутри объекта',
            'mean_object_probability': 'Среднее grayscale-значение внутри объекта',
            'uncertain_fraction': 'Доля сомнительных пикселей',
            'object_area_fraction': 'Доля площади объекта',
            'mean_point_confidence': 'Средняя уверенность по точкам',
            'mean_point_probability': 'Среднее grayscale-значение по точкам',
            'mean_point_contrast': 'Средний контраст точек',
            'point_count': 'Количество точек',
            'soft_dice': 'Soft Dice',
            'soft_iou': 'Soft IoU',
            'ssim': 'SSIM',
            'dice': 'Dice',
            'iou': 'IoU',
            'hausdorff_distance': 'Расстояние Хаусдорфа',
            'centroid_distance': 'Расстояние между центроидами',
            'mae': 'MAE',
            'rmse': 'RMSE',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1',
            'f1_at_r': 'F1@r',
            'tp': 'TP',
            'fp': 'FP',
            'fn': 'FN',
            'bce': 'BCE',
            'iou_score': 'Score IoU',
            'dice_score': 'Score Dice',
            'polygon_bce_score': 'Score BCE',
            'overall_polygon_score': 'Итоговый score полигонов',
            'precision_score': 'Score Precision',
            'recall_score': 'Score Recall',
            'f1_score': 'Score F1',
            'overall_point_score': 'Итоговый score точек',
            'mean_localization_distance': 'Средняя ошибка локализации',
            'mean_localization_error': 'Средняя ошибка локализации',
            'localization_score': 'Оценка локализации',
            'localization_agreement': 'Согласованность локализации',
            'count_error': 'Ошибка количества',
            'count_agreement': 'Согласованность количества',
            'connected_component_error': 'Ошибка числа компонент',
            'cc_error': 'Ошибка числа компонент',
            'chamfer_score': 'Оценка Chamfer',
            'hausdorff_score': 'Оценка Хаусдорфа',
        }
        labels = labels_ru if getattr(self._i18n, 'language', 'en') == 'ru' else labels_en
        if name in labels:
            return labels[name]
        return name.replace('_', ' ')

    def _component_value_text(self, value: str) -> str:
        values_en = {
            'labeled frame': 'labeled frame',
            'unlabeled frame': 'unlabeled frame',
            'supervised error map': 'supervised error map',
        }
        values_ru = {
            'labeled frame': 'размеченный кадр',
            'unlabeled frame': 'неразмеченный кадр',
            'supervised error map': 'supervised-карта ошибки',
            'variance/entropy risk map': 'карта риска по variance / entropy',
            'mean entropy of consensus probability': 'средняя энтропия consensus probability',
            'mean variance over model probability maps': 'средняя дисперсия probability maps моделей',
            'acquisition_score': 'приоритет на разметку',
        }
        values = values_ru if getattr(self._i18n, 'language', 'en') == 'ru' else values_en
        return values.get(value, value.replace(' vs ', ' против ') if getattr(self._i18n, 'language', 'en') == 'ru' else value)

    def _decorate_metric_lines(self, metric_key: str, summary, lines: list[str]) -> list[str]:
        decorated: list[str] = []
        hint = self._metric_hint(metric_key, summary)
        if hint:
            decorated.append(hint)
        for line in lines:
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]
            status_prefix = ''
            for status in ('active', 'auxiliary', 'legacy'):
                prefix = f'{status} '
                if stripped.startswith(prefix):
                    status_prefix = f"{self._t(f'status.{status}')} "
                    stripped = stripped[len(prefix):]
                    break
            if ':' in stripped:
                name, value = stripped.split(':', 1)
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

    def _percentile_map_for_metric(self, state: ExtendMatrixTabState, metric_key: str) -> dict[str, float]:
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

    def _repeated_percentile_entries(self, state: ExtendMatrixTabState, *, band: str) -> list[tuple[FrameRecord, int, float, list[str]]]:
        available_keys = list(self._percentile_basis_keys_for_state(state, state.build_result))
        cache_key = (str(band), tuple(str(key) for key in available_keys), id(state.build_result.records))
        cached = state.repeated_percentile_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        metrics_by_record: dict[str, list[tuple[str, float]]] = {record.key: [] for record in state.build_result.records}
        for metric_key in available_keys:
            percentile_map = self._percentile_map_for_metric(state, metric_key)
            for record in state.build_result.records:
                percentile = percentile_map.get(record.key)
                if percentile is None:
                    continue
                if band == 'bad' and float(percentile) < 15.0:
                    metrics_by_record[record.key].append((self._metric_label(metric_key, state.build_result), float(percentile)))
                elif band == 'good' and float(percentile) >= 60.0:
                    metrics_by_record[record.key].append((self._metric_label(metric_key, state.build_result), float(percentile)))
        entries: list[tuple[FrameRecord, int, float, list[str]]] = []
        records_by_key = {record.key: record for record in state.build_result.records}
        for key, values in metrics_by_record.items():
            if not values:
                continue
            metric_labels = [label for label, _percentile in values]
            average_percentile = sum(percentile for _label, percentile in values) / float(len(values))
            record = records_by_key[key]
            entries.append((record, len(values), average_percentile, metric_labels))
        if band == 'bad':
            entries.sort(key=lambda item: (-item[1], item[2], item[0].display_name.lower()))
        else:
            entries.sort(key=lambda item: (-item[1], -item[2], item[0].display_name.lower()))
        state.repeated_percentile_cache[cache_key] = tuple(entries)
        return entries

    def _update_repeated_percentile_lists(self, state: ExtendMatrixTabState) -> None:
        def summary(entries: list[tuple[FrameRecord, int, float, list[str]]]) -> tuple[int, float, int]:
            visible_entries = entries[:25]
            if not visible_entries:
                return 0, 0.0, 0
            frame_count = len(visible_entries)
            mean_hits = sum(count for _record, count, _avg_percentile, _labels in visible_entries) / float(frame_count)
            max_hits = max(count for _record, count, _avg_percentile, _labels in visible_entries)
            return frame_count, mean_hits, max_hits

        bad_entries = self._repeated_percentile_entries(state, band='bad')
        good_entries = self._repeated_percentile_entries(state, band='good')
        if state.repeated_bad_column is not None and hasattr(state.repeated_bad_column, 'set_payload'):
            frame_count, mean_hits, max_hits = summary(bad_entries)
            state.repeated_bad_column.set_payload(frame_count, mean_hits, max_hits, active=state.correlation_filter_band == 'bad')
        if state.repeated_good_column is not None and hasattr(state.repeated_good_column, 'set_payload'):
            frame_count, mean_hits, max_hits = summary(good_entries)
            state.repeated_good_column.set_payload(frame_count, mean_hits, max_hits, active=state.correlation_filter_band == 'good')

    def _update_metric_histograms(self, state: ExtendMatrixTabState) -> None:
        preview = state.preview
        if preview is None:
            return
        base_record_count = len(self._base_records_for_state(state))
        available_keys = set(self._percentile_basis_keys_for_state(state, state.build_result))
        if state.percentile_filter_metric_key not in available_keys:
            state.percentile_filter_metric_key = None
            state.percentile_filter_bin_index = None
        for metric_key, card in preview.histogram_cards.items():
            visible = metric_key in available_keys
            if not visible:
                card.setVisible(False)
                continue
            counts = self._percentile_histogram_counts(state, metric_key)
            active_bin = state.percentile_filter_bin_index if state.percentile_filter_metric_key == metric_key else None
            tooltip = self._metric_hint_fallback(metric_key, state.build_result)
            card.set_payload(self._metric_label(metric_key, state.build_result), counts, base_record_count, visible=True, active_bin=active_bin, tooltip=tooltip)
        self._update_repeated_percentile_lists(state)

    def _schedule_metric_histogram_update(self, state: ExtendMatrixTabState) -> None:
        self._histogram_update_generation += 1
        generation = int(self._histogram_update_generation)
        QTimer.singleShot(0, lambda s=state, g=generation: self._update_metric_histograms_chunked(s, g, 0))

    def _update_metric_histograms_chunked(self, state: ExtendMatrixTabState, generation: int, index: int) -> None:
        if generation != self._histogram_update_generation or state.widget not in self._tab_states:
            return
        preview = state.preview
        if preview is None:
            return
        base_record_count = len(self._base_records_for_state(state))
        available_keys = set(self._percentile_basis_keys_for_state(state, state.build_result))
        if state.percentile_filter_metric_key not in available_keys:
            state.percentile_filter_metric_key = None
            state.percentile_filter_bin_index = None
        histogram_items = list(preview.histogram_cards.items())
        if index >= len(histogram_items):
            QTimer.singleShot(0, lambda s=state, g=generation: self._update_repeated_percentile_lists(s) if g == self._histogram_update_generation and s.widget in self._tab_states else None)
            return
        metric_key, card = histogram_items[index]
        if metric_key not in available_keys:
            card.setVisible(False)
        else:
            counts = self._percentile_histogram_counts(state, metric_key)
            active_bin = state.percentile_filter_bin_index if state.percentile_filter_metric_key == metric_key else None
            tooltip = self._metric_hint_fallback(metric_key, state.build_result)
            card.set_payload(self._metric_label(metric_key, state.build_result), counts, base_record_count, visible=True, active_bin=active_bin, tooltip=tooltip)
        QTimer.singleShot(0, lambda s=state, g=generation, i=index + 1: self._update_metric_histograms_chunked(s, g, i))

    def _connect_histogram_cards(self, state: ExtendMatrixTabState) -> None:
        preview = state.preview
        if preview is None:
            return
        for metric_key, card in preview.histogram_cards.items():
            if hasattr(card, 'binClicked'):
                card.binClicked.connect(lambda clicked_metric_key, bin_index, s=state: self._on_histogram_bin_clicked(s, str(clicked_metric_key), int(bin_index)))
        if state.repeated_bad_column is not None:
            state.repeated_bad_column.columnClicked.connect(lambda band, s=state: self._on_correlation_column_clicked(s, str(band)))
        if state.repeated_good_column is not None:
            state.repeated_good_column.columnClicked.connect(lambda band, s=state: self._on_correlation_column_clicked(s, str(band)))

    def _on_correlation_column_clicked(self, state: ExtendMatrixTabState, band: str) -> None:
        state.percentile_filter_metric_key = None
        state.percentile_filter_bin_index = None
        state.correlation_filter_band = None if state.correlation_filter_band == band else str(band)
        self._apply_tab_visual_settings(state, reset_view=False)
        if state.content_tabs is not None:
            state.content_tabs.setCurrentIndex(0)

    def _on_histogram_bin_clicked(self, state: ExtendMatrixTabState, metric_key: str, bin_index: int) -> None:
        same_filter = state.percentile_filter_metric_key == metric_key and state.percentile_filter_bin_index == int(bin_index)
        state.correlation_filter_band = None
        if same_filter:
            state.percentile_filter_metric_key = None
            state.percentile_filter_bin_index = None
        else:
            state.percentile_filter_metric_key = str(metric_key)
            state.percentile_filter_bin_index = int(bin_index)
        self._apply_tab_visual_settings(state, reset_view=False)

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
        if metric_key in {"overall_frame_score", "export_priority_score"}:
            if summary.labeled_best_quality is not None:
                return [
                    "source: labeled frame",
                    "formula: 1 - labeled_best_quality",
                    f"labeled_best_quality: {summary.labeled_best_quality:.4f}",
                ]
            return [
                "source: unlabeled frame",
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
                    lines.append(f"  active mean_localization_error: {float(row.get('mean_localization_error', 0.0)):.4f}")
                    lines.append(f"  active localization_agreement: {float(row.get('localization_agreement', 0.0)):.4f}")
                    lines.append(f"  active count_agreement: {float(row.get('count_agreement', 0.0)):.4f}")
            return lines or [f"disagreement_score: {summary.disagreement_score:.4f}"]
        if metric_key == "disagreement_score":
            return [
                "formula: 1 - model_model_score",
                f"model_model_score: {summary.metric_values.get('model_model_score', 0.0):.4f}",
            ]
        if metric_key in {"overall_polygon_score", "iou_score", "dice_score", "polygon_bce_score", "iou", "dice", "bce"}:
            return [
                f"iou: {self._format_component_value(summary.metric_values.get('iou'))}",
                f"dice: {self._format_component_value(summary.metric_values.get('dice'))}",
                f"bce: {self._format_component_value(summary.metric_values.get('bce'))}",
                f"iou_score: {self._format_component_value(summary.metric_values.get('iou_score'))}",
                f"dice_score: {self._format_component_value(summary.metric_values.get('dice_score'))}",
                f"polygon_bce_score: {self._format_component_value(summary.metric_values.get('polygon_bce_score'))}",
                f"overall_polygon_score: {self._format_component_value(summary.metric_values.get('overall_polygon_score'))}",
            ]
        if metric_key in {"overall_point_score", "precision_score", "recall_score", "f1_score", "localization_score", "precision", "recall", "f1", "mean_localization_distance"}:
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
        parsed_metric = metric_key.split('::', 1) if '::' in str(metric_key) else None
        if parsed_metric is not None:
            family, model_id = parsed_metric
            confidence_row = summary.model_confidence.get(model_id) if summary.model_confidence is not None else None
            model_name = self._model_display_name(state, model_id)
            if family == 'model_confidence' and confidence_row is not None:
                if hasattr(confidence_row, 'mean_object_confidence'):
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
            if family == 'model_uncertain_fraction' and confidence_row is not None:
                return [
                    f"model: {model_name}",
                    f"uncertain_fraction: {self._format_component_value(getattr(confidence_row, 'uncertain_fraction', None))}",
                    f"mean_boundary_uncertainty: {self._format_component_value(getattr(confidence_row, 'mean_boundary_uncertainty', None))}",
                    f"mean_transition_width: {self._format_component_value(getattr(confidence_row, 'mean_transition_width', None))}",
                    f"mean_core_confidence: {self._format_component_value(getattr(confidence_row, 'mean_core_confidence', None))}",
                ]
            if family == 'model_point_contrast' and confidence_row is not None:
                return [
                    f"model: {model_name}",
                    f"mean_point_contrast: {self._format_component_value(getattr(confidence_row, 'mean_point_contrast', None))}",
                    f"mean_local_confidence: {self._format_component_value(getattr(confidence_row, 'mean_local_confidence', None))}",
                    f"mean_center_confidence: {self._format_component_value(getattr(confidence_row, 'mean_center_confidence', None))}",
                    f"point_count: {self._format_component_value(getattr(confidence_row, 'point_count', None))}",
                ]
        if metric_key in {"model_labeled_score", "labeled_best_quality", "labeled_mean_quality"}:
            lines = []
            for spec in state.build_result.model_specs:
                metrics = summary.model_metrics.get(spec.model_id)
                if metrics is None:
                    continue
                score = getattr(metrics, "quality_score", None)
                if score is None:
                    continue
                lines.append(f"{spec.display_name}: {float(score):.4f}")
                if summary.frame_type == "polygon" and hasattr(metrics, "soft_dice"):
                    lines.append(f"  active soft_dice: {self._format_component_value(getattr(metrics, 'soft_dice', None))}")
                    lines.append(f"  active soft_iou: {self._format_component_value(getattr(metrics, 'soft_iou', None))}")
                    lines.append(f"  active ssim: {self._format_component_value(getattr(metrics, 'ssim', None))}")
                    lines.append(f"  active dice: {self._format_component_value(getattr(metrics, 'dice', None))}")
                    lines.append(f"  active iou: {self._format_component_value(getattr(metrics, 'iou', None))}")
                    lines.append(f"  active hausdorff_distance: {self._format_component_value(getattr(metrics, 'hausdorff_distance', None))}")
                    lines.append(f"  active centroid_distance: {self._format_component_value(getattr(metrics, 'centroid_distance', None))}")
                    lines.append(f"  auxiliary mae: {self._format_component_value(getattr(metrics, 'mae', None))}")
                    lines.append(f"  auxiliary rmse: {self._format_component_value(getattr(metrics, 'rmse', None))}")
                    lines.append(f"  auxiliary precision: {self._format_component_value(getattr(metrics, 'precision', None))}")
                    lines.append(f"  auxiliary recall: {self._format_component_value(getattr(metrics, 'recall', None))}")
                    lines.append(f"  auxiliary count_error: {self._format_component_value(getattr(metrics, 'count_error', None))}")
                    lines.append(f"  auxiliary cc_error: {self._format_component_value(getattr(metrics, 'connected_component_error', None))}")
                else:
                    lines.append(f"  active precision: {self._format_component_value(getattr(metrics, 'precision_at_radius', None))}")
                    lines.append(f"  active recall: {self._format_component_value(getattr(metrics, 'recall_at_radius', None))}")
                    lines.append(f"  active f1_at_r: {self._format_component_value(getattr(metrics, 'f1_at_radius', None))}")
                    lines.append(f"  active mean_localization_error: {self._format_component_value(getattr(metrics, 'mean_localization_error', None))}")
                    lines.append(f"  active localization_score: {self._format_component_value(getattr(metrics, 'localization_score', None))}")
                    lines.append(f"  active count_error: {self._format_component_value(getattr(metrics, 'count_error', None))}")
                    lines.append(f"  auxiliary chamfer_score: {self._format_component_value(getattr(metrics, 'chamfer_score', None))}")
                    lines.append(f"  auxiliary hausdorff_score: {self._format_component_value(getattr(metrics, 'hausdorff_score', None))}")
            return lines
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

    def _show_progress_bar(self, *, visible: bool, current: int = 0, total: int = 0, key: str = "", format_text: str | None = None) -> None:
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
        gt_text, gt_tooltip = self._compact_folder_label(self._gt_folder)
        self.original_folder_value.setText(original_text)
        self.original_folder_value.setToolTip(original_tooltip)
        self.gt_folder_value.setText(gt_text)
        self.gt_folder_value.setToolTip(gt_tooltip)

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
        self._sync_action_buttons()

    def _clear_folders(self) -> None:
        self.folder_list.clear()
        self._refresh_folder_rows()
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

    def _set_gt_folder(self) -> None:
        if self._worker_thread is not None:
            return
        folder = QFileDialog.getExistingDirectory(self._view, self._t("dialog.select_gt_folder"))
        if not folder:
            return
        path = Path(folder)
        if not self._folder_has_supported_images(path):
            QMessageBox.warning(
                self._view,
                self._t("dialog.warning_title"),
                f"Labeled folder has no supported images: {path}",
            )
            return
        self._gt_folder = FolderSpec(path=path, label=path.name)
        self._update_source_labels()
        self._sync_action_buttons()

    def _clear_gt_folder(self) -> None:
        self._gt_folder = None
        self._update_source_labels()
        self._sync_action_buttons()

    def _start_build(self) -> None:
        model_specs = self._checked_model_specs()
        if not model_specs:
            QMessageBox.warning(self._view, self._t("dialog.warning_title"), self._t("errors.active_model_required"))
            return
        self._close_all_details_dialogs()
        geometry_mode = geometry_mode_for_object_type(self._selected_object_type())
        mask_threshold, boundary_radius = self._selected_polygon_compare_values()
        options = BuildOptions(
            thumbnail_size=int(DEFAULT_CELL_SIZE),
            recursive=True,
            tile_mode=self._selected_tile_mode(),
            tile_width=int(self.tile_width_spin.value()),
            tile_height=int(self.tile_height_spin.value()),
            tile_overlap_mode=self._selected_tile_overlap_mode(),
            tile_overlap=int(self.tile_overlap_spin.value()),
            subpixel_view_mode=self._selected_subpixel_view_mode(),
            subpixel_rows=self._selected_subpixel_rows(),
            subpixel_columns=self._selected_subpixel_columns(),
            subpixel_aggregation=self._selected_subpixel_aggregation(),
            geometry_mode=geometry_mode,
            mask_threshold=float(mask_threshold),
            boundary_radius=int(boundary_radius),
            confidence_uncertainty_delta=self._selected_confidence_uncertainty_delta(),
            point_match_radius=float(self.point_match_radius_spin.value()),
            point_confidence_radius=int(self.point_confidence_radius_spin.value()),
            point_extraction_mode=str(self.point_extraction_mode_combo.currentData() or DEFAULT_POINT_EXTRACTION_MODE),
            polygon_confidence_summary=str(self.polygon_confidence_summary_combo.currentData() or DEFAULT_POLYGON_CONFIDENCE_SUMMARY),
        )
        self._pending_build_snapshot = self._capture_view_snapshot()
        self._worker_kind = "build"
        self._worker_thread = QThread(self._view)
        self._worker = FrameIndexWorker(model_specs, options, self._original_folder, self._gt_folder)
        generation = self._begin_worker_request(state=None)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda current, total, key, g=generation: self._on_build_progress(current, total, key, generation=g))
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

    def _open_record_details(self, record: FrameRecord, state: ExtendMatrixTabState, tile_selection: object | None = None) -> None:
        session_view_state = dict(self._details_view_payload)
        if tile_selection is not None:
            parent_row = int(getattr(tile_selection, "matrix_row", getattr(tile_selection, "row", 0)))
            parent_column = int(getattr(tile_selection, "matrix_column", getattr(tile_selection, "column", 0)))
            sub_row = int(getattr(tile_selection, "sub_row", getattr(tile_selection, "tile_row", getattr(tile_selection, "row", 0))))
            sub_column = int(getattr(tile_selection, "sub_column", getattr(tile_selection, "tile_column", getattr(tile_selection, "column", 0))))
            spec = getattr(tile_selection, "spec", None)
            session_view_state["subpixel_selection"] = {
                "parent_row": parent_row,
                "parent_column": parent_column,
                "sub_row": sub_row,
                "sub_column": sub_column,
                "parent_value": float(getattr(tile_selection, "parent_value", getattr(record, "score", 0.0))),
                "subpixel_value": float(getattr(tile_selection, "subpixel_value", getattr(record, "score", 0.0))),
                "subpixel_confidence": None if getattr(tile_selection, "subpixel_confidence", None) is None else float(getattr(tile_selection, "subpixel_confidence")),
                "aggregation": str(getattr(tile_selection, "aggregation", "mean")),
                "metric_key": str(getattr(tile_selection, "metric_key", state.metric_key or DEFAULT_MATRIX_METRIC_KEY)),
                "spec_rows": int(getattr(spec, "rows", 0) or 0),
                "spec_columns": int(getattr(spec, "columns", 0) or 0),
            }
            session_view_state["tile_selection"] = {
                "row": sub_row,
                "column": sub_column,
            }
        else:
            session_view_state.pop("tile_selection", None)
            session_view_state.pop("subpixel_selection", None)
        dialog = ExtendFrameDetailsDialog(
            record=record,
            build_result=state.build_result,
            preferred_metric_key=state.metric_key,
            session_view_state=session_view_state,
            on_view_state_changed=self._store_details_view_payload,
            parent=None,
        )
        dialog.setModal(False)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.destroyed.connect(lambda *_args, dialog=dialog: self._forget_details_dialog(dialog))
        self._details_dialogs.append(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _store_details_view_payload(self, payload: dict[str, object]) -> None:
        self._details_view_payload = dict(payload or {})
        self._settings_service.save_details_view_payload(self._details_view_payload)

    def _update_matrix_preview(self, state: ExtendMatrixTabState, record: FrameRecord | None = None) -> None:
        selected = record or state.matrix_view.current_record()
        preview = state.preview
        if preview is None:
            return
        if selected is None:
            preview.frame_value.setText("-")
            if preview.subpixel_group is not None:
                preview.subpixel_group.hide()
            if preview.subpixel_value is not None:
                preview.subpixel_value.setText("-")
            if getattr(preview, "subpixel_score_card", None) is not None:
                preview.subpixel_score_card.hide()
            for card in preview.score_cards.values():
                card.set_payload("-", self._metric_score_style(None, state.metric_key), "", visible=False, percentile_text="", percentile_style=self._percentile_style(None))
            if preview.overall_group is not None:
                preview.overall_group.hide()
            if preview.component_group is not None:
                preview.component_group.hide()
            return
        preview.frame_value.setText(selected.display_name)
        summary = selected.summary
        selected_subpixel = None
        if hasattr(state.matrix_view, "selected_tile_selection"):
            try:
                selected_subpixel = state.matrix_view.selected_tile_selection()
            except Exception:
                selected_subpixel = None
        if (
            preview.subpixel_group is not None
            and preview.subpixel_value is not None
            and selected_subpixel is not None
            and getattr(selected_subpixel, "record", None) is not None
            and getattr(selected_subpixel.record, "key", None) == selected.key
        ):
            parent_row = int(getattr(selected_subpixel, "matrix_row", getattr(selected_subpixel, "row", 0))) + 1
            parent_column = int(getattr(selected_subpixel, "matrix_column", getattr(selected_subpixel, "column", 0))) + 1
            sub_row = int(getattr(selected_subpixel, "sub_row", 0)) + 1
            sub_column = int(getattr(selected_subpixel, "sub_column", 0)) + 1
            spec = getattr(selected_subpixel, "spec", None)
            rows = int(getattr(spec, "rows", 0) or 0)
            columns = int(getattr(spec, "columns", 0) or 0)
            parent_value = float(getattr(selected_subpixel, "parent_value", selected.score if bool(getattr(selected, "score_ready", False)) else 0.0))
            subpixel_value = float(getattr(selected_subpixel, "subpixel_value", parent_value))
            summary = selected.summary
            frame_type = str(getattr(summary, "frame_type", "") or self._t("details.frame_type"))
            preview.subpixel_value.setText(
                self._t(
                    "details.subpixel_selection_value",
                    parent_row=parent_row,
                    parent_column=parent_column,
                    sub_row=sub_row,
                    sub_column=sub_column,
                    rows=rows,
                    columns=columns,
                    parent_value=parent_value,
                )
            )
            preview.subpixel_group.setTitle(self._t("details.subpixel_selection"))
            preview.subpixel_value.show()
            preview.subpixel_group.show()
            if getattr(preview, "subpixel_score_card", None) is not None:
                subpixel_metric_key = str(state.metric_key or state.build_result.selected_metric_key or DEFAULT_MATRIX_METRIC_KEY)
                subpixel_title = self._metric_text_for_key(subpixel_metric_key, state.build_result)
                subpixel_details = "\n".join(
                    [
                        f"{self._t('details.frame_type')}: {frame_type}",
                        f"{self._t('details.selected_comparison_score')}: {self._metric_score_text(subpixel_value, subpixel_metric_key)}",
                        f"{self._t('details.subpixel_selection')}: {subpixel_value:.4f}",
                        f"{self._t('details.parent_score')}: {float(getattr(selected_subpixel, 'parent_value', selected.score if bool(getattr(selected, 'score_ready', False)) else 0.0)):.4f}",
                    ]
                )
                preview.subpixel_score_card.set_payload(
                    self._metric_score_text(subpixel_value, subpixel_metric_key),
                    self._metric_score_style(subpixel_value, subpixel_metric_key),
                    subpixel_details,
                    visible=True,
                    percentile_text="",
                    percentile_style=self._percentile_style(None),
                    tooltip=self._metric_hint(subpixel_metric_key, selected.summary) if selected.summary is not None else self._metric_hint_fallback(subpixel_metric_key, state.build_result),
                )
                try:
                    preview.subpixel_score_card.title_label.setText(subpixel_title)
                except Exception:
                    pass
        elif preview.subpixel_group is not None:
            preview.subpixel_group.hide()
            if preview.subpixel_value is not None:
                preview.subpixel_value.setText("-")
            if getattr(preview, "subpixel_score_card", None) is not None:
                preview.subpixel_score_card.hide()
        percentile_cache: dict[str, dict[str, float]] = {}
        visible_metric_keys = set(self._display_metric_keys_for_state(state, state.build_result))
        overall_visible = False
        component_visible = False
        for metric_key, card in preview.score_cards.items():
            value = metric_value_for_record(selected, metric_key) if summary is not None else None
            visible = metric_key in visible_metric_keys and value is not None
            details = "\n".join(self._decorate_metric_lines(metric_key, summary, self._metric_component_lines(state, selected, metric_key))) if visible else ""
            percentile_map = percentile_cache.setdefault(metric_key, self._percentile_map_for_metric(state, metric_key)) if visible else {}
            percentile_value = percentile_map.get(selected.key) if visible else None
            tooltip = self._metric_hint(metric_key, summary) if summary is not None else self._metric_hint_fallback(metric_key, state.build_result)
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
        is_busy = self._worker_thread is not None
        self.btn_clear_folders.setEnabled(self.folder_list.count() > 0 and not is_busy)
        self.btn_set_original.setEnabled(not is_busy)
        self.btn_clear_original.setEnabled(self._original_folder is not None and not is_busy)
        self.btn_set_gt.setEnabled(not is_busy)
        self.btn_clear_gt.setEnabled(self._gt_folder is not None and not is_busy)
        self.btn_build.setEnabled(active_model_count > 0 and not is_busy)
        self.btn_compute.setEnabled(current_state is not None and not is_busy)
        self.btn_cancel.setEnabled(is_busy)
        if hasattr(self._view, "set_workflow_summary"):
            original_state = self._t("workflow.state.ready") if self._original_folder is not None else self._t("workflow.state.pending")
            gt_state = self._t("workflow.state.ready") if self._gt_folder is not None else self._t("workflow.state.optional")
            sources_tone = "ready" if self._original_folder is not None else "warn"
            models_status = self._t("workflow.state.ready") if active_model_count > 0 else self._t("workflow.state.pending")
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
            self._view.set_workflow_summary({
                "sources": (
                    self._t("workflow.state.partial") if self._original_folder is None and self._gt_folder is not None else original_state,
                    self._t("workflow.sources_detail", original=original_state, gt=gt_state),
                    sources_tone,
                ),
                "models": (
                    models_status,
                    self._t("workflow.models_detail", count=active_model_count),
                    models_tone,
                ),
                "analysis": (analysis_status, analysis_detail, analysis_tone),
            })

    def _build_folder_manager_payload(self) -> dict:
        return {
            "folders": [
                {
                    "path": str(self.folder_list.item(row).data(Qt.ItemDataRole.UserRole)),
                    "checked": bool(self.folder_list.item(row).data(FOLDER_CHECKED_ROLE)),
                    "label": str(self.folder_list.item(row).data(FOLDER_LABEL_ROLE) or ""),
                    "confidence_path": str(self.folder_list.item(row).data(FOLDER_CONFIDENCE_ROLE) or ""),
                }
                for row in range(self.folder_list.count())
            ],
            "original_folder": str(self._original_folder.path) if self._original_folder is not None else None,
            "gt_folder": str(self._gt_folder.path) if self._gt_folder is not None else None,
        }

    def _restore_persisted_state(self) -> None:
        self._restore_folder_manager_state()
        self._restore_build_settings()
        self._update_source_labels()

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
            original_folder = payload.get("original_folder")
            gt_folder = payload.get("gt_folder")
            if original_folder and Path(original_folder).exists():
                path = Path(original_folder)
                self._original_folder = FolderSpec(path=path, label=path.name)
            if gt_folder and Path(gt_folder).exists():
                path = Path(gt_folder)
                self._gt_folder = FolderSpec(path=path, label=path.name)
        finally:
            self.folder_list.blockSignals(False)

    def _build_build_settings_payload(self) -> dict:
        mask_threshold, boundary_radius = self._selected_polygon_compare_values()
        return {
            "thumbnail_size": int(DEFAULT_CELL_SIZE),
            "matrix_score_view_mode": str(self.matrix_score_view_combo.currentData() or DEFAULT_MATRIX_SCORE_VIEW_MODE),
            "analysis_mode": self._selected_analysis_mode(),
            "object_type": self._selected_object_type(),
            "geometry_mode": str(self.geometry_mode_combo.currentData() or DEFAULT_GEOMETRY_MODE),
            "polygon_compare_profile": self._selected_polygon_compare_profile(),
            "mask_threshold": float(mask_threshold),
            "boundary_radius": int(boundary_radius),
            "confidence_uncertainty_profile": self._selected_confidence_uncertainty_profile(),
            "confidence_uncertainty_delta": self._selected_confidence_uncertainty_delta(),
            "point_match_radius": float(self.point_match_radius_spin.value()),
            "point_confidence_radius": int(self.point_confidence_radius_spin.value()),
            "point_extraction_mode": str(self.point_extraction_mode_combo.currentData() or DEFAULT_POINT_EXTRACTION_MODE),
            "polygon_confidence_summary": str(self.polygon_confidence_summary_combo.currentData() or DEFAULT_POLYGON_CONFIDENCE_SUMMARY),
            "tile_mode": self._selected_tile_mode(),
            "tile_width": int(self.tile_width_spin.value()),
            "tile_height": int(self.tile_height_spin.value()),
            "tile_overlap_mode": self._selected_tile_overlap_mode(),
            "tile_overlap": int(self.tile_overlap_spin.value()),
            "subpixel_view_mode": self._selected_subpixel_view_mode(),
            "subpixel_rows": self._selected_subpixel_rows(),
            "subpixel_columns": self._selected_subpixel_columns(),
            "subpixel_aggregation": self._selected_subpixel_aggregation(),
            "layout_mode": str(self.layout_mode_combo.currentData() or DEFAULT_MATRIX_LAYOUT_MODE),
            "total_frames": int(self.total_frames_spin.value()),
            "frames_per_row": int(self.frames_per_row_spin.value()),
            "rows": int(self.matrix_rows_spin.value()),
            "columns": int(self.matrix_columns_spin.value()),
            "metric_key": str(self.metric_combo.currentData() or DEFAULT_MATRIX_METRIC_KEY),
            "metric_scope": str(self.metric_scope_combo.currentData() or ""),
            "confidence_model_id": str(self.metric_scope_combo.currentData() or ""),
            "frame_type_filter": str(self.frame_type_filter_combo.currentData() or 'all'),
        }

    def _restore_build_settings(self) -> None:
        payload = self._settings_service.load_build_settings_payload() or {}
        blockers = [
            QSignalBlocker(self.thumbnail_size_spin),
            QSignalBlocker(self.matrix_score_view_combo),
            QSignalBlocker(self.analysis_mode_combo),
            QSignalBlocker(self.geometry_mode_combo),
            QSignalBlocker(self.polygon_compare_profile_combo),
            QSignalBlocker(self.mask_threshold_spin),
            QSignalBlocker(self.boundary_radius_spin),
            QSignalBlocker(self.confidence_uncertainty_profile_combo),
            QSignalBlocker(self.point_match_radius_spin),
            QSignalBlocker(self.point_confidence_radius_spin),
            QSignalBlocker(self.point_extraction_mode_combo),
            QSignalBlocker(self.tile_mode_combo),
            QSignalBlocker(self.subpixel_view_mode_combo),
            QSignalBlocker(self.subpixel_rows_spin),
            QSignalBlocker(self.subpixel_columns_spin),
            QSignalBlocker(self.subpixel_aggregation_combo),
            QSignalBlocker(self.tile_width_spin),
            QSignalBlocker(self.tile_height_spin),
            QSignalBlocker(self.tile_overlap_mode_combo),
            QSignalBlocker(self.tile_overlap_spin),
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
        analysis_mode = str(payload.get("analysis_mode") or self._selected_analysis_mode())
        analysis_index = self.analysis_mode_combo.findData(analysis_mode)
        self.analysis_mode_combo.setCurrentIndex(analysis_index if analysis_index >= 0 else 0)
        geometry_mode = str(payload.get("geometry_mode") or geometry_mode_for_object_type(payload.get("object_type")).value or DEFAULT_GEOMETRY_MODE)
        geometry_index = self.geometry_mode_combo.findData(geometry_mode)
        self.geometry_mode_combo.setCurrentIndex(geometry_index if geometry_index >= 0 else 0)
        tile_mode = str(payload.get("tile_mode") or DEFAULT_TILE_MODE)
        if tile_mode == "subpixel":
            tile_mode = "tile"
        tile_mode_index = self.tile_mode_combo.findData(tile_mode)
        self.tile_mode_combo.setCurrentIndex(tile_mode_index if tile_mode_index >= 0 else 0)
        subpixel_view_mode = str(payload.get("subpixel_view_mode") or DEFAULT_SUBPIXEL_VIEW_MODE)
        if subpixel_view_mode == "subpixel":
            subpixel_view_mode = "tile"
        subpixel_view_mode_index = self.subpixel_view_mode_combo.findData(subpixel_view_mode)
        self.subpixel_view_mode_combo.setCurrentIndex(subpixel_view_mode_index if subpixel_view_mode_index >= 0 else 0)
        self.subpixel_rows_spin.setValue(int(payload.get("subpixel_rows", DEFAULT_SUBPIXEL_ROWS)))
        self.subpixel_columns_spin.setValue(int(payload.get("subpixel_columns", DEFAULT_SUBPIXEL_COLUMNS)))
        subpixel_aggregation = str(payload.get("subpixel_aggregation") or DEFAULT_SUBPIXEL_AGGREGATION)
        subpixel_aggregation_index = self.subpixel_aggregation_combo.findData(subpixel_aggregation)
        self.subpixel_aggregation_combo.setCurrentIndex(subpixel_aggregation_index if subpixel_aggregation_index >= 0 else 0)
        self.tile_width_spin.setValue(int(payload.get("tile_width", DEFAULT_TILE_WIDTH)))
        self.tile_height_spin.setValue(int(payload.get("tile_height", DEFAULT_TILE_HEIGHT)))
        tile_overlap_mode = str(payload.get("tile_overlap_mode") or DEFAULT_TILE_OVERLAP_MODE)
        tile_overlap_mode_index = self.tile_overlap_mode_combo.findData(tile_overlap_mode)
        self.tile_overlap_mode_combo.setCurrentIndex(tile_overlap_mode_index if tile_overlap_mode_index >= 0 else 0)
        self.tile_overlap_spin.setValue(int(payload.get("tile_overlap", DEFAULT_TILE_OVERLAP)))
        compare_profile = str(payload.get("polygon_compare_profile") or "")
        self.mask_threshold_spin.setValue(float(payload.get("mask_threshold", self.mask_threshold_spin.value())))
        self.boundary_radius_spin.setValue(int(payload.get("boundary_radius", self.boundary_radius_spin.value())))
        if not compare_profile:
            compare_profile = self._polygon_compare_profile_for_values(self.mask_threshold_spin.value(), self.boundary_radius_spin.value())
        compare_index = self.polygon_compare_profile_combo.findData(compare_profile)
        self.polygon_compare_profile_combo.setCurrentIndex(compare_index if compare_index >= 0 else 0)
        uncertainty_profile = str(payload.get("confidence_uncertainty_profile") or "")
        if not uncertainty_profile:
            uncertainty_profile = self._confidence_uncertainty_profile_for_value(payload.get("confidence_uncertainty_delta"))
        uncertainty_index = self.confidence_uncertainty_profile_combo.findData(uncertainty_profile)
        self.confidence_uncertainty_profile_combo.setCurrentIndex(uncertainty_index if uncertainty_index >= 0 else 0)
        self.point_match_radius_spin.setValue(float(payload.get("point_match_radius", self.point_match_radius_spin.value())))
        self.point_confidence_radius_spin.setValue(int(payload.get("point_confidence_radius", DEFAULT_POINT_CONFIDENCE_RADIUS)))
        point_extraction_mode = str(payload.get("point_extraction_mode") or DEFAULT_POINT_EXTRACTION_MODE)
        point_mode_index = self.point_extraction_mode_combo.findData(point_extraction_mode)
        self.point_extraction_mode_combo.setCurrentIndex(point_mode_index if point_mode_index >= 0 else 0)
        polygon_confidence_summary = str(payload.get("polygon_confidence_summary") or DEFAULT_POLYGON_CONFIDENCE_SUMMARY)
        polygon_summary_index = self.polygon_confidence_summary_combo.findData(polygon_confidence_summary)
        self.polygon_confidence_summary_combo.setCurrentIndex(polygon_summary_index if polygon_summary_index >= 0 else 0)
        layout_mode = str(payload.get("layout_mode") or DEFAULT_MATRIX_LAYOUT_MODE)
        layout_index = self.layout_mode_combo.findData(layout_mode)
        self.layout_mode_combo.setCurrentIndex(layout_index if layout_index >= 0 else 0)
        self.total_frames_spin.setValue(int(payload.get("total_frames", DEFAULT_TOTAL_FRAMES)))
        self.frames_per_row_spin.setValue(int(payload.get("frames_per_row", DEFAULT_FRAMES_PER_ROW)))
        self.matrix_rows_spin.setValue(int(payload.get("rows", DEFAULT_MATRIX_ROWS)))
        self.matrix_columns_spin.setValue(int(payload.get("columns", DEFAULT_MATRIX_COLUMNS)))
        metric_key = str(payload.get("metric_key") or DEFAULT_MATRIX_METRIC_KEY)
        metric_scope = str(payload.get("confidence_model_id") or payload.get("metric_scope") or self._metric_scope_for_metric_key(metric_key) or "")
        frame_type_filter = str(payload.get('frame_type_filter') or self._selected_object_type())
        self._sync_metric_controls(None, preferred_metric_key=metric_key, preferred_scope_key=metric_scope)
        index = self.frame_type_filter_combo.findData(frame_type_filter)
        self.frame_type_filter_combo.setCurrentIndex(index if index >= 0 else 0)
        self._sync_tile_overlap_bounds()
        self._sync_mode_controls(None, None)

    def _persist_state(self) -> None:
        self._settings_service.save_folder_manager_payload(self._build_folder_manager_payload())
        self._settings_service.save_build_settings_payload(self._build_build_settings_payload())
        self._settings_service.sync()

    def shutdown(self) -> None:
        if self._worker is not None:
            request_cancel = getattr(self._worker, "request_cancel", None)
            if callable(request_cancel):
                request_cancel()
        thread = self._worker_thread
        if thread is not None:
            try:
                thread.quit()
            except Exception:
                pass
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


# Backward-compatible alias for legacy lite imports.
ValidationGradientLitePresenter = ValidationGradientExtendPresenter


