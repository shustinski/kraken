"""Main window for the extended validation gradient widget."""

from __future__ import annotations
from PyQt6.QtCore import QSettings, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QMenuBar,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QStyle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)
from kraken_core.theme import apply_app_theme, normalize_theme
from ..infra.services import ValidationGradientLiteSettingsService
from ..core.analysis_modes import ANALYSIS_MODE_OPTIONS, default_confidence_model_id
from ..ui.i18n import Translator, set_current_language
from ..ui.matrix_view import MatrixListWidget, MatrixMiniMapWidget
from ..ui.ui_constants import (
    BOUNDARY_RADIUS_RANGE,
    CONFIDENCE_UNCERTAINTY_PROFILE_OPTIONS,
    CONTROL_PANEL_SPLITTER_SIZES,
    DEFAULT_BOUNDARY_RADIUS,
    DEFAULT_CELL_SIZE,
    DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE,
    DEFAULT_ANALYSIS_MODE,
    DEFAULT_EXPORT_NEIGHBOR_RADIUS,
    DEFAULT_EXPORT_PERCENT,
    DEFAULT_EXPORT_PERCENTILE,
    DEFAULT_EXPORT_SELECTION_MODE,
    DEFAULT_FILTER_TO_EXPORT_CANDIDATES,
    DEFAULT_FRAMES_PER_ROW,
    DEFAULT_GEOMETRY_MODE,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_POLYGON_COMPARE_PROFILE,
    DEFAULT_POINT_CONFIDENCE_RADIUS,
    DEFAULT_POINT_EXTRACTION_MODE,
    DEFAULT_POINT_MATCH_RADIUS,
    DEFAULT_POLYGON_CONFIDENCE_SUMMARY,
    DEFAULT_MATRIX_COLUMNS,
    DEFAULT_MATRIX_LAYOUT_MODE,
    DEFAULT_MATRIX_SCORE_VIEW_MODE,
    DEFAULT_MATRIX_METRIC_KEY,
    DEFAULT_METRIC_SCOPE,
    DEFAULT_MATRIX_ROWS,
    DEFAULT_SUBPIXEL_AGGREGATION,
    DEFAULT_SUBPIXEL_COLUMNS,
    DEFAULT_SUBPIXEL_ROWS,
    DEFAULT_SUBPIXEL_VIEW_MODE,
    DEFAULT_TILE_HEIGHT,
    DEFAULT_TILE_MODE,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_OVERLAP_MODE,
    DEFAULT_TILE_WIDTH,
    DEFAULT_TOP_K_EXPORT,
    DEFAULT_TOTAL_FRAMES,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    EXPORT_NEIGHBOR_RANGE,
    EXPORT_PERCENTILE_RANGE,
    EXPORT_PERCENT_RANGE,
    EXPORT_SELECTION_MODE_OPTIONS,
    EXPORT_TOP_K_RANGE,
    EXTEND_LANGUAGE_BUTTON_OBJECT_NAME,
    EXTEND_ROOT_OBJECT_NAME,
    EXTEND_WIDGET_STYLESHEET,
    FRAMES_PER_ROW_RANGE,
    GEOMETRY_MODE_OPTIONS,
    MASK_THRESHOLD_RANGE,
    MATRIX_COLUMNS_RANGE,
    METRIC_SETTINGS_COMBO_MIN_CONTENTS_LENGTH,
    METRIC_SETTINGS_LABEL_MIN_WIDTH,
    MATRIX_METRIC_GROUP_OPTIONS,
    MATRIX_METRIC_OPTIONS,
    MATRIX_SCORE_VIEW_OPTIONS,
    MATRIX_ROWS_RANGE,
    SUBPIXEL_AGGREGATION_OPTIONS,
    SUBPIXEL_COLUMNS_RANGE,
    SUBPIXEL_ROWS_RANGE,
    SUBPIXEL_VIEW_MODE_OPTIONS,
    OVERVIEW_PANEL_MAX_WIDTH,
    PERCENTILE_BAND_BOUNDS,
    PERCENTILE_BAND_COLORS,
    PERCENTILE_BAND_LABELS,
    PERCENTILE_BAND_TITLES,
    POINT_CONFIDENCE_RADIUS_RANGE,
    POINT_MATCH_RADIUS_RANGE,
    POLYGON_CONFIDENCE_SUMMARY_OPTIONS,
    POLYGON_COMPARE_PROFILE_OPTIONS,
    SETTINGS_APP,
    SETTINGS_LABEL_MIN_WIDTH,
    SETTINGS_ORG,
    TILE_MODE_OPTIONS,
    TILE_OVERLAP_MODE_OPTIONS,
    TILE_OVERLAP_RANGE,
    TILE_SIZE_RANGE,
    THUMBNAIL_SIZE_RANGE,
    TOTAL_FRAMES_RANGE,
)
from .presenter import ValidationGradientExtendPresenter
from .state import ExtendMatrixTabState, ExtendPreviewPanel

from .main_window_parts.construction import ConstructionMixin
from .main_window_parts.build_ui import BuildUiMixin
from .main_window_parts.metric_text_key import MetricTextKeyMixin
from .main_window_parts.populate_tile_mode_combo import PopulateTileModeComboMixin
from .main_window_parts.build_matrix_settings_widget import BuildMatrixSettingsWidgetMixin
from .main_window_parts.build_metric_settings_widget import BuildMetricSettingsWidgetMixin
from .main_window_parts.retranslate_ui import RetranslateUiMixin
from .main_window_parts.build_setting_row import BuildSettingRowMixin
from .main_window_parts.build_preview_panel import BuildPreviewPanelMixin
from .main_window_parts.connect_signals import ConnectSignalsMixin
from .main_window_parts.create_matrix_tab import CreateMatrixTabMixin
from .main_window_parts.close_event import CloseEventMixin
from .main_window_parts.expandable_score_card import (
    _ExpandableScoreCard,
    _NoWheelSpinBox,
    _NoWheelDoubleSpinBox,
    _NoWheelComboBox,
)
from .main_window_parts.percentile_histogram_widget import (
    _PercentileHistogramWidget,
    _PercentileHistogramCard,
)
from .main_window_parts.correlation_column_widget import _CorrelationColumnWidget
from .main_window_parts.validation_gradient_extend_main import ValidationGradientExtendMainWindow

class ValidationGradientExtendWidget(
    ConstructionMixin,
    BuildUiMixin,
    MetricTextKeyMixin,
    PopulateTileModeComboMixin,
    BuildMatrixSettingsWidgetMixin,
    BuildMetricSettingsWidgetMixin,
    RetranslateUiMixin,
    BuildSettingRowMixin,
    BuildPreviewPanelMixin,
    ConnectSignalsMixin,
    CreateMatrixTabMixin,
    CloseEventMixin,
    QWidget,
):
    """Embeddable widget for multi-model segmentation quality evaluation."""

# Backward-compatible aliases for legacy lite imports.
ValidationGradientLiteWidget = ValidationGradientExtendWidget
ValidationGradientLiteMainWindow = ValidationGradientExtendMainWindow

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _ExpandableScoreCard,
    _NoWheelSpinBox,
    _NoWheelDoubleSpinBox,
    _NoWheelComboBox,
    _PercentileHistogramWidget,
    _PercentileHistogramCard,
    _CorrelationColumnWidget,
    ValidationGradientExtendMainWindow,
)
