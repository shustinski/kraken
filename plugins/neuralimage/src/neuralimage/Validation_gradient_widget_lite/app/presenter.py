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

from .presenter_parts.sync_mode_controls import SyncModeControlsMixin
from .presenter_parts.update_subpixel_plan_label import UpdateSubpixelPlanLabelMixin
from .presenter_parts.refresh_folder_rows import RefreshFolderRowsMixin
from .presenter_parts.ui_context_state import UiContextStateMixin
from .presenter_parts.sync_metric_controls import SyncMetricControlsMixin
from .presenter_parts.metric_score_style import MetricScoreStyleMixin
from .presenter_parts.metric_label import MetricLabelMixin
from .presenter_parts.metric_hint_fallback import MetricHintFallbackMixin
from .presenter_parts.metric_component_summary import MetricComponentSummaryMixin
from .presenter_parts.component_name_label import ComponentNameLabelMixin
from .presenter_parts.repeated_percentile_entries import RepeatedPercentileEntriesMixin
from .presenter_parts.connect_histogram_cards import ConnectHistogramCardsMixin
from .presenter_parts.metric_component_lines import MetricComponentLinesMixin
from .presenter_parts.start_build import StartBuildMixin
from .presenter_parts.start_compute_analytics import StartComputeAnalyticsMixin
from .presenter_parts.apply_metric_state import ApplyMetricStateMixin
from .presenter_parts.open_record_details import OpenRecordDetailsMixin
from .presenter_parts.update_matrix_preview import UpdateMatrixPreviewMixin
from .presenter_parts.sync_action_buttons import SyncActionButtonsMixin
from .presenter_parts.restore_build_settings import RestoreBuildSettingsMixin
from .presenter_parts.shutdown import ShutdownMixin

class ValidationGradientExtendPresenter(
    SyncModeControlsMixin,
    UpdateSubpixelPlanLabelMixin,
    RefreshFolderRowsMixin,
    UiContextStateMixin,
    SyncMetricControlsMixin,
    MetricScoreStyleMixin,
    MetricLabelMixin,
    MetricHintFallbackMixin,
    MetricComponentSummaryMixin,
    ComponentNameLabelMixin,
    RepeatedPercentileEntriesMixin,
    ConnectHistogramCardsMixin,
    MetricComponentLinesMixin,
    StartBuildMixin,
    StartComputeAnalyticsMixin,
    ApplyMetricStateMixin,
    OpenRecordDetailsMixin,
    UpdateMatrixPreviewMixin,
    SyncActionButtonsMixin,
    RestoreBuildSettingsMixin,
    ShutdownMixin,
    QObject,
):
    """Coordinate UI state, background workers and matrix tabs."""

# Backward-compatible alias for legacy lite imports.
ValidationGradientLitePresenter = ValidationGradientExtendPresenter
