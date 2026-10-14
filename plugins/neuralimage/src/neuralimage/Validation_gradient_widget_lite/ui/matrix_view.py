"""Implement mismatch-only matrix layout, rendering and overview widgets for the lite tool."""

from __future__ import annotations
import math
import hashlib
import pickle
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPaintEvent, QPen, QPixmap, QTransform
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QToolTip,
    QVBoxLayout,
    QWidget,
)
from ..core.backend_constants import CACHE_DIR, FRAME_NUMBER_PATTERN
from ..core.analysis_modes import confidence_metric_family, metric_level_key, metric_visual_ratio
from ..core.domain import FrameRecord
from ..core.repository import (
    POLYGON_SUPPORT_THRESHOLD,
    _frame_uncertainty_components_from_probability,
    compute_comparison_score,
    load_frame_layers,
    load_grayscale_image,
)
from ..core.domain import ComparisonMode
from ..core.subpixel_grid import (
    SubpixelGrid,
    SubpixelGridSpec,
    SubpixelSelection,
    aggregate_subpixel_values,
    build_subpixel_grid_from_array,
    build_subpixel_grid_from_pair,
)
from .i18n import Translator
from ..core.repository import metric_higher_is_better, natural_sort_key
from .ui_constants import (
    CARD_CONTENT_SPACING,
    DEFAULT_BORDER,
    DEFAULT_CELL_SIZE,
    DEFAULT_ERROR_WINDOW,
    DEFAULT_GRADIENT_NAME,
    GRADIENT_LABELS,
    GRADIENT_PRESETS,
    GRADIENT_PREVIEW_MIN_HEIGHT,
    GRADIENT_RANGE_SELECTOR_MIN_HEIGHT,
    HOVER_BORDER,
    MATRIX_BACKGROUND,
    MATRIX_BACKGROUND_ALT,
    MATRIX_CELL_GAP,
    MATRIX_DEFAULT_PEN_WIDTH,
    MATRIX_HOVER_PEN_WIDTH,
    MATRIX_MAX_SCALE,
    MATRIX_MIN_CELL_SIZE,
    MATRIX_MIN_SCALE,
    MATRIX_PROCESSING_PEN_WIDTH,
    MATRIX_REFERENCE_PEN_WIDTH,
    MATRIX_SCENE_PADDING,
    MATRIX_SELECTED_BLEND_RATIO,
    MINIMAP_FRAME_MARGIN,
    MINIMAP_MIN_SIZE,
    MINIMAP_PROCESSING_TRIANGLE_HALF_WIDTH,
    MINIMAP_PROCESSING_TRIANGLE_HEIGHT,
    MINIMAP_REFERENCE_MARKER_SIDE,
    MINIMAP_REFERENCE_PEN_WIDTH,
    MINIMAP_SELECTED_COLOR,
    MINIMAP_SELECTED_OUTLINE_WIDTH,
    MINIMAP_SELECTED_RADIUS_OFF,
    MINIMAP_SELECTED_RADIUS_ON,
    NORMALIZATION_EPSILON,
    PANEL_BACKGROUND,
    PANEL_TEXT,
    PROCESSING_BORDER,
    PROCESSING_FILL,
    REFERENCE_BORDER,
    SELECTION_BLINK_INTERVAL_MS,
    SELECTED_BLINK_COLOR,
    SUBDUED_TEXT_COLOR,
    VISIBLE_RECT_MIN_SIZE,
)

from .matrix_view_parts.module_state import (
    SUBPIXEL_VISIBILITY_THRESHOLD,
    MIN_VISIBLE_TILE_SCREEN_SIZE,
    LOW_ZOOM_OVERVIEW_MAX_ZOOM,
    LOW_ZOOM_OVERVIEW_RECORD_THRESHOLD,
    VIEW_LOD_OVERVIEW,
    VIEW_LOD_PIXEL,
    VIEW_LOD_SUBPIXEL,
    TILE_VIEWPORT_DEBOUNCE_MS,
    TILE_HOVER_PREFETCH_MS,
    TILE_LOAD_SLICE_BUDGET_MS,
    TILE_LOAD_MAX_PER_SLICE,
    TILE_LOAD_MAX_KEYS_PER_SLICE,
    TILE_PREFETCH_MARGIN_CELLS,
    SUBPIXEL_GRID_CACHE_MAX_ITEMS,
    SUBPIXEL_GRID_DISK_CACHE_DIR,
    SUBPIXEL_GRID_DISK_CACHE_VERSION,
    SUBPIXEL_GRID_DISK_CACHE_MAX_FILES,
    SUBPIXEL_GRID_DISK_CACHE_TRIM_INTERVAL_SECONDS,
    SUBPIXEL_GRID_WORKER_COUNT,
    SUBPIXEL_GRID_MAX_IN_FLIGHT,
    MATRIX_VIRTUALIZE_RECORD_THRESHOLD,
    MATRIX_ITEM_KEEP_MARGIN_CELLS,
    MATRIX_MAX_MATERIALIZED_ITEMS,
    SUBPIXEL_VISIBILITY_EXIT_THRESHOLD,
    _subpixel_grid_disk_cache_last_trim,
)
from .matrix_view_parts.construction import ConstructionMixin
from .matrix_view_parts.subpixel_grid_record import SubpixelGridRecordMixin
from .matrix_view_parts.refresh_scene import RefreshSceneMixin
from .matrix_view_parts.prioritized_record_keys import PrioritizedRecordKeysMixin
from .matrix_view_parts.prepare_visible_tile_queue import PrepareVisibleTileQueueMixin
from .matrix_view_parts.process_pending_tile_queue import ProcessPendingTileQueueMixin
from .matrix_view_parts.poll_tile_futures import PollTileFuturesMixin
from .matrix_view_parts.mouse_release_event import MouseReleaseEventMixin
from .matrix_view_parts.rebuild_scene import RebuildSceneMixin
from .matrix_view_parts.emit_overview_state import EmitOverviewStateMixin
from .matrix_view_parts.tile_selection_cell import TileSelectionCellMixin
from .matrix_view_parts.subpixel_color_value import SubpixelColorValueMixin
from .matrix_view_parts.score_fill_color import (
    MatrixLayoutConfig,
    MatrixTileSelection,
    _tile_rect_for_cell,
    _display_tile_index_for_cell,
    _tile_screen_extent_for_rect,
    _subpixel_overlay_visible_for_rect,
    _contrast_text_color,
    _score_fill_color,
    _path_cache_identity,
    _record_path_signature,
)
from .matrix_view_parts.store_subpixel_grid_disk import (
    _subpixel_cache_key_for_record,
    _subpixel_disk_cache_path,
    _load_subpixel_grid_from_disk,
    _trim_subpixel_grid_disk_cache,
    _store_subpixel_grid_to_disk,
)
from .matrix_view_parts.build_subpixel_grid_record import _build_subpixel_grid_for_record
from .matrix_view_parts.matrix_cell_item import _MatrixCellItem
from .matrix_view_parts.compute_auto_color_window import (
    extract_frame_number,
    blend_colors,
    interpolate_gradient_color,
    error_palette_color,
    map_score_to_palette_position,
    compute_auto_color_window,
    enhance_palette_position,
)
from .matrix_view_parts.build_matrix_layout import build_matrix_layout
from .matrix_view_parts.gradient_preview_bar import _GradientPreviewBar
from .matrix_view_parts.gradient_window_bar import _GradientWindowBar
from .matrix_view_parts.gradient_preset_selector_widget import (
    _GradientPresetCard,
    GradientPresetSelectorWidget,
)
from .matrix_view_parts.gradient_range_selector_widget import GradientRangeSelectorWidget
from .matrix_view_parts.matrix_mini_map_widget import MatrixMiniMapWidget

class MatrixListWidget(
    ConstructionMixin,
    SubpixelGridRecordMixin,
    RefreshSceneMixin,
    PrioritizedRecordKeysMixin,
    PrepareVisibleTileQueueMixin,
    ProcessPendingTileQueueMixin,
    PollTileFuturesMixin,
    MouseReleaseEventMixin,
    RebuildSceneMixin,
    EmitOverviewStateMixin,
    TileSelectionCellMixin,
    SubpixelColorValueMixin,
    QGraphicsView,
):
    """Render the frame matrix and provide navigation, selection and overview data."""
    recordActivated = pyqtSignal(object)
    recordSelected = pyqtSignal(object)
    tileSelected = pyqtSignal(object)
    tileActivated = pyqtSignal(object)
    contextMenuRequested = pyqtSignal(object, object)
    overviewChanged = pyqtSignal(object, object, object, object, object, object)

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    SUBPIXEL_VISIBILITY_THRESHOLD,
    MIN_VISIBLE_TILE_SCREEN_SIZE,
    LOW_ZOOM_OVERVIEW_MAX_ZOOM,
    LOW_ZOOM_OVERVIEW_RECORD_THRESHOLD,
    VIEW_LOD_OVERVIEW,
    VIEW_LOD_PIXEL,
    VIEW_LOD_SUBPIXEL,
    TILE_VIEWPORT_DEBOUNCE_MS,
    TILE_HOVER_PREFETCH_MS,
    TILE_LOAD_SLICE_BUDGET_MS,
    TILE_LOAD_MAX_PER_SLICE,
    TILE_LOAD_MAX_KEYS_PER_SLICE,
    TILE_PREFETCH_MARGIN_CELLS,
    SUBPIXEL_GRID_CACHE_MAX_ITEMS,
    SUBPIXEL_GRID_DISK_CACHE_DIR,
    SUBPIXEL_GRID_DISK_CACHE_VERSION,
    SUBPIXEL_GRID_DISK_CACHE_MAX_FILES,
    SUBPIXEL_GRID_DISK_CACHE_TRIM_INTERVAL_SECONDS,
    SUBPIXEL_GRID_WORKER_COUNT,
    SUBPIXEL_GRID_MAX_IN_FLIGHT,
    MATRIX_VIRTUALIZE_RECORD_THRESHOLD,
    MATRIX_ITEM_KEEP_MARGIN_CELLS,
    MATRIX_MAX_MATERIALIZED_ITEMS,
    SUBPIXEL_VISIBILITY_EXIT_THRESHOLD,
    _subpixel_grid_disk_cache_last_trim,
    MatrixLayoutConfig,
    MatrixTileSelection,
    _tile_rect_for_cell,
    _display_tile_index_for_cell,
    _tile_screen_extent_for_rect,
    _subpixel_overlay_visible_for_rect,
    _contrast_text_color,
    _score_fill_color,
    _path_cache_identity,
    _record_path_signature,
    _subpixel_cache_key_for_record,
    _subpixel_disk_cache_path,
    _load_subpixel_grid_from_disk,
    _trim_subpixel_grid_disk_cache,
    _store_subpixel_grid_to_disk,
    _build_subpixel_grid_for_record,
    _MatrixCellItem,
    extract_frame_number,
    blend_colors,
    interpolate_gradient_color,
    error_palette_color,
    map_score_to_palette_position,
    compute_auto_color_window,
    enhance_palette_position,
    build_matrix_layout,
    _GradientPreviewBar,
    _GradientWindowBar,
    _GradientPresetCard,
    GradientPresetSelectorWidget,
    GradientRangeSelectorWidget,
    MatrixMiniMapWidget,
)

__all__ = [
    "GradientPresetSelectorWidget",
    "GradientRangeSelectorWidget",
    "compute_auto_color_window",
    "enhance_palette_position",
    "MatrixLayoutConfig",
    "MatrixTileSelection",
    "MatrixListWidget",
    "MatrixMiniMapWidget",
    "blend_colors",
    "build_matrix_layout",
    "error_palette_color",
    "extract_frame_number",
    "interpolate_gradient_color",
    "map_score_to_palette_position",
]
