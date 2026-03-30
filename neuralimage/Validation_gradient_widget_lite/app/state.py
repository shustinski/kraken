"""Store lightweight per-tab state used by the lite widget."""
from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6.QtWidgets import QGroupBox, QLabel, QWidget

from ..core.domain import BuildResult
from ..ui.matrix_view import MatrixLayoutConfig, MatrixListWidget, MatrixMiniMapWidget
from ..ui.ui_constants import DEFAULT_CELL_SIZE, DEFAULT_ERROR_WINDOW, DEFAULT_GRADIENT_NAME, DEFAULT_SCORE_VIEW_MODE


@dataclass(slots=True)
class LitePreviewPanel:
    """Store the selected-frame preview widgets shown next to one matrix tab."""

    group: QGroupBox
    frame_title: QLabel
    frame_value: QLabel
    absolute_title: QLabel
    absolute_value: QLabel
    relative_title: QLabel
    relative_value: QLabel


@dataclass(slots=True)
class LiteMatrixTabState:
    """Store the UI objects and mutable mismatch-only state for one matrix tab."""

    widget: QWidget
    matrix_view: MatrixListWidget
    mini_map: MatrixMiniMapWidget
    build_result: BuildResult
    cell_size: int = DEFAULT_CELL_SIZE
    layout_config: MatrixLayoutConfig = field(default_factory=MatrixLayoutConfig)
    gradient_name: str = DEFAULT_GRADIENT_NAME
    error_window: tuple[float, float] = DEFAULT_ERROR_WINDOW
    score_view_mode: str = DEFAULT_SCORE_VIEW_MODE
    preview: LitePreviewPanel | None = None

