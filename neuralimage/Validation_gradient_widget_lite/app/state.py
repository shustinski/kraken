"""UI state containers for the extended validation gradient widget."""
from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6.QtWidgets import QGroupBox, QLabel, QTabWidget, QWidget

from ..core.analysis_modes import INTER_MODEL_ANALYSIS_MODE, POLYGON_OBJECT_TYPE
from ..core.domain import BuildResult
from ..ui.matrix_view import MatrixLayoutConfig, MatrixListWidget, MatrixMiniMapWidget
from ..ui.ui_constants import DEFAULT_CELL_SIZE, DEFAULT_MATRIX_METRIC_KEY, DEFAULT_METRIC_SCOPE, DEFAULT_MATRIX_SCORE_VIEW_MODE


@dataclass(slots=True)
class ExtendPreviewPanel:
    """Store the selected-frame preview widgets shown next to one matrix tab."""

    group: QGroupBox
    frame_title: QLabel
    frame_value: QLabel
    subpixel_group: QGroupBox | None = None
    subpixel_value: QLabel | None = None
    subpixel_score_card: QWidget | None = None
    overall_group: QGroupBox | None = None
    component_group: QGroupBox | None = None
    score_cards: dict[str, QWidget] = field(default_factory=dict)
    histogram_cards: dict[str, QWidget] = field(default_factory=dict)


@dataclass(slots=True)
class ExtendMatrixTabState:
    """Store the UI objects and mutable state for one matrix tab."""

    widget: QWidget
    matrix_view: MatrixListWidget
    mini_map: MatrixMiniMapWidget
    build_result: BuildResult
    content_tabs: QTabWidget | None = None
    cell_size: int = DEFAULT_CELL_SIZE
    layout_config: MatrixLayoutConfig = field(default_factory=MatrixLayoutConfig)
    matrix_score_view_mode: str = DEFAULT_MATRIX_SCORE_VIEW_MODE
    metric_key: str = DEFAULT_MATRIX_METRIC_KEY
    metric_scope: str = DEFAULT_METRIC_SCOPE
    analysis_mode: str = INTER_MODEL_ANALYSIS_MODE
    object_type: str = POLYGON_OBJECT_TYPE
    confidence_model_id: str | None = None
    frame_type_filter: str = 'all'
    preview: ExtendPreviewPanel | None = None
    percentile_filter_metric_key: str | None = None
    percentile_filter_bin_index: int | None = None
    correlation_filter_band: str | None = None
    percentile_cache: dict[tuple[str, tuple[str, ...]], dict[str, float]] = field(default_factory=dict)
    metric_result_cache: dict[str, BuildResult] = field(default_factory=dict)
    base_records_cache: dict[tuple[str, int], tuple] = field(default_factory=dict)
    repeated_percentile_cache: dict[tuple[str, tuple[str, ...], int], tuple] = field(default_factory=dict)
    processing_state_by_key: dict[str, str] = field(default_factory=dict)
    repeated_bad_column: QWidget | None = None
    repeated_good_column: QWidget | None = None


# Backward-compatible aliases for legacy lite imports.
LitePreviewPanel = ExtendPreviewPanel
LiteMatrixTabState = ExtendMatrixTabState

