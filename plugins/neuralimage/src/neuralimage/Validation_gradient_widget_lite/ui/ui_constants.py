"""UI constants for the extended validation gradient widget."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from ..core.analysis_modes import INTER_MODEL_ANALYSIS_MODE
from ..core.domain import ComparisonMode, GeometryMode

SETTINGS_ORG = "ValidationGradientExcend"
SETTINGS_APP = "ValidationGradientExcend"
SETTINGS_FOLDERS_KEY = "ui/model_folders"
SETTINGS_BUILD_KEY = "ui/build_settings"
SETTINGS_DETAILS_VIEW_KEY = "ui/details_view_settings"
SETTINGS_LANGUAGE_KEY = "ui/language"
SETTINGS_ORIGINAL_FOLDER_KEY = "ui/original_folder"
SETTINGS_GT_FOLDER_KEY = "ui/gt_folder"

FOLDER_CHECKED_ROLE = int(Qt.ItemDataRole.UserRole) + 1
FOLDER_LABEL_ROLE = int(Qt.ItemDataRole.UserRole) + 2
FOLDER_CONFIDENCE_ROLE = int(Qt.ItemDataRole.UserRole) + 3

DEFAULT_COMPARISON_MODE = ComparisonMode.DISAGREEMENT
DEFAULT_CELL_SIZE = 15
DEFAULT_GRADIENT_NAME = "traffic_lights"
DEFAULT_ERROR_WINDOW = (0.0, 1.0)
DEFAULT_MATRIX_METRIC_KEY = "overall_frame_score"
DEFAULT_ANALYSIS_MODE = INTER_MODEL_ANALYSIS_MODE
DEFAULT_METRIC_SCOPE = ""
DEFAULT_MATRIX_LAYOUT_MODE = "indexed_grid"
DEFAULT_MATRIX_SCORE_VIEW_MODE = "relative"
DEFAULT_TILE_MODE = "pixel"
DEFAULT_SUBPIXEL_VIEW_MODE = "pixel"
DEFAULT_SUBPIXEL_ROWS = 2
DEFAULT_SUBPIXEL_COLUMNS = 2
DEFAULT_SUBPIXEL_AGGREGATION = "mean"
DEFAULT_TILE_WIDTH = 256
DEFAULT_TILE_HEIGHT = 256
DEFAULT_TILE_OVERLAP_MODE = "auto"
DEFAULT_TILE_OVERLAP = 0
DEFAULT_TOTAL_FRAMES = 10000
DEFAULT_FRAMES_PER_ROW = 100
DEFAULT_MATRIX_ROWS = 100
DEFAULT_MATRIX_COLUMNS = 100
DEFAULT_TOP_K_EXPORT = 32
DEFAULT_EXPORT_PERCENT = 10
DEFAULT_EXPORT_PERCENTILE = 90
DEFAULT_EXPORT_NEIGHBOR_RADIUS = 1
DEFAULT_EXPORT_SELECTION_MODE = "count"
DEFAULT_FILTER_TO_EXPORT_CANDIDATES = False
DEFAULT_GEOMETRY_MODE = GeometryMode.MASK.value
DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_BOUNDARY_RADIUS = 1
DEFAULT_POLYGON_COMPARE_PROFILE = "balanced"
DEFAULT_CONFIDENCE_UNCERTAINTY_DELTA = 0.10
DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE = "standard"
DEFAULT_POINT_MATCH_RADIUS = 3.0
DEFAULT_POINT_CONFIDENCE_RADIUS = 3
DEFAULT_POINT_EXTRACTION_MODE = 'component_centroids'
DEFAULT_POLYGON_CONFIDENCE_SUMMARY = 'weighted'
DEFAULT_WINDOW_WIDTH = 1500
DEFAULT_WINDOW_HEIGHT = 920
NORMALIZATION_EPSILON = 1e-9

TOTAL_FRAMES_RANGE = (0, 10_000_000)
FRAMES_PER_ROW_RANGE = (0, 100_000)
MATRIX_ROWS_RANGE = (1, 100_000)
MATRIX_COLUMNS_RANGE = (1, 100_000)
TILE_SIZE_RANGE = (1, 100_000)
TILE_OVERLAP_RANGE = (0, 100_000)
SUBPIXEL_ROWS_RANGE = (1, 64)
SUBPIXEL_COLUMNS_RANGE = (1, 64)
THUMBNAIL_SIZE_RANGE = (2, 128)
BOUNDARY_RADIUS_RANGE = (1, 16)
POINT_MATCH_RADIUS_RANGE = (1.0, 64.0)
POINT_CONFIDENCE_RADIUS_RANGE = (1, 16)
MASK_THRESHOLD_RANGE = (0.0, 1.0)
TILE_MODE_OPTIONS = (
    ("matrix.tile_mode.pixel", "pixel"),
    ("matrix.tile_mode.subpixel", "subpixel"),
)
SUBPIXEL_VIEW_MODE_OPTIONS = (
    ("matrix.subpixel_view_mode.pixel", "pixel"),
    ("matrix.subpixel_view_mode.tile", "tile"),
)
SUBPIXEL_AGGREGATION_OPTIONS = (
    ("subpixel_aggregation.mean", "mean"),
    ("subpixel_aggregation.weighted_mean", "weighted_mean"),
    ("subpixel_aggregation.median", "median"),
)
TILE_OVERLAP_MODE_OPTIONS = (
    ("matrix.tile_overlap_mode.auto", "auto"),
    ("matrix.tile_overlap_mode.manual", "manual"),
)
POLYGON_COMPARE_PROFILE_OPTIONS = (
    ("polygon_compare_profile.lenient", "lenient"),
    ("polygon_compare_profile.balanced", "balanced"),
    ("polygon_compare_profile.strict", "strict"),
)
POLYGON_COMPARE_PROFILE_VALUES = {
    "lenient": (0.45, 1),
    "balanced": (0.50, 1),
    "strict": (0.60, 2),
}
EXPORT_TOP_K_RANGE = (1, 10000)
EXPORT_PERCENT_RANGE = (1, 100)
EXPORT_PERCENTILE_RANGE = (1, 100)
EXPORT_NEIGHBOR_RANGE = (0, 16)
CONTROL_PANEL_SPLITTER_SIZES = (400, 1100)
SETTINGS_LABEL_MIN_WIDTH = 120
METRIC_SETTINGS_LABEL_MIN_WIDTH = 140
METRIC_SETTINGS_COMBO_MIN_CONTENTS_LENGTH = 14
OVERVIEW_PANEL_MAX_WIDTH = 380
FOLDER_BUTTON_SIZE = 20
FOLDER_ROW_MIN_HEIGHT = 58
CARD_CONTENT_SPACING = 3
GRADIENT_PREVIEW_MIN_HEIGHT = 22
GRADIENT_RANGE_SELECTOR_MIN_HEIGHT = 60

MATRIX_CELL_GAP = 1
MATRIX_SCENE_PADDING = 4
MATRIX_MIN_CELL_SIZE = 2
MATRIX_MIN_SCALE = 0.05
MATRIX_MAX_SCALE = 64.0
SELECTION_BLINK_INTERVAL_MS = 400
MATRIX_SELECTED_BLEND_RATIO = 0.45
VISIBLE_RECT_MIN_SIZE = 2.0
MATRIX_DEFAULT_PEN_WIDTH = 0.5
MATRIX_HOVER_PEN_WIDTH = 1.2
MATRIX_PROCESSING_PEN_WIDTH = 1.4
MATRIX_REFERENCE_PEN_WIDTH = 1.8

PERCENTILE_BAND_BOUNDS = (
    (0.0, 15.0),
    (15.0, 35.0),
    (35.0, 60.0),
    (60.0, 100.0),
)
PERCENTILE_BAND_LABELS = ("0-15", "15-35", "35-60", "60-100")
PERCENTILE_BAND_TITLES = ("P0-15", "P15-35", "P35-60", "P60-100")
PERCENTILE_BAND_COLORS = ("#8c2f39", "#a75d12", "#6f7a18", "#1f5f3b")

MINIMAP_MIN_SIZE = (160, 160)
MINIMAP_FRAME_MARGIN = 6
MINIMAP_PROCESSING_TRIANGLE_HALF_WIDTH = 2.0
MINIMAP_PROCESSING_TRIANGLE_HEIGHT = 2.0
MINIMAP_REFERENCE_MARKER_SIDE = 3.0
MINIMAP_REFERENCE_PEN_WIDTH = 1.2
MINIMAP_SELECTED_OUTLINE_WIDTH = 1.5
MINIMAP_SELECTED_RADIUS_ON = 4.0
MINIMAP_SELECTED_RADIUS_OFF = 2.5

PANEL_BACKGROUND = QColor(37, 37, 38)
PANEL_TEXT = QColor(235, 235, 235)
SUBDUED_TEXT_COLOR = QColor(120, 120, 120)
MATRIX_BACKGROUND = QColor(30, 30, 30)
MATRIX_BACKGROUND_ALT = QColor(52, 52, 52)
DEFAULT_BORDER = QColor(70, 70, 70)
HOVER_BORDER = QColor(255, 220, 120)
PROCESSING_BORDER = QColor(255, 170, 0)
PROCESSING_FILL = QColor(255, 170, 0)
REFERENCE_BORDER = QColor(80, 210, 255)
SELECTED_BLINK_COLOR = QColor(255, 255, 255)
MINIMAP_SELECTED_COLOR = QColor(255, 255, 255)

GRADIENT_PRESETS = {
    # One standardized matrix palette: bad -> red, good -> bright green.
    "traffic_lights": (
        (0.0, (186, 0, 0)),
        (0.18, (233, 74, 0)),
        (0.42, (255, 196, 0)),
        (0.68, (168, 228, 0)),
        (1.0, (0, 255, 96)),
    ),
}
GRADIENT_LABELS = {"traffic_lights": "Standard"}

GEOMETRY_MODE_OPTIONS = (
    (GeometryMode.MASK.label, GeometryMode.MASK.value),
    (GeometryMode.POINT.label, GeometryMode.POINT.value),
    (GeometryMode.AUTO.label, GeometryMode.AUTO.value),
)

POLYGON_CONFIDENCE_SUMMARY_OPTIONS = (
    ('polygon_confidence_summary.weighted', 'weighted'),
    ('polygon_confidence_summary.core', 'core'),
)

CONFIDENCE_UNCERTAINTY_PROFILE_OPTIONS = (
    ("confidence_delta_profile.soft", "soft"),
    ("confidence_delta_profile.standard", "standard"),
    ("confidence_delta_profile.strict", "strict"),
)

CONFIDENCE_UNCERTAINTY_PROFILE_VALUES = {
    "soft": 0.06,
    "standard": 0.10,
    "strict": 0.15,
}

EXPORT_SELECTION_MODE_OPTIONS = (
    ("Worst frame count", "count"),
    ("Worst percent of frames", "percent"),
    ("Badness percentile threshold", "percentile"),
)

MATRIX_METRIC_GROUP_OPTIONS = (
    ("metric.group.overall", "overall"),
    ("metric.group.model_model", "model_model"),
    ("metric.group.model_labeled", "model_labeled"),
)

MATRIX_METRIC_OPTIONS = (
    ("metric.overall_frame_score", "overall_frame_score", "overall"),
    ("metric.export_priority_score", "export_priority_score", "overall"),
    ("metric.model_model_score", "model_model_score", "model_model"),
    ("metric.disagreement_score", "disagreement_score", "model_model"),
    ("metric.model_labeled_score", "model_labeled_score", "model_labeled"),
    ("metric.labeled_best_quality", "labeled_best_quality", "model_labeled"),
    ("metric.labeled_mean_quality", "labeled_mean_quality", "model_labeled"),
)

MATRIX_SCORE_VIEW_OPTIONS = (
    ("matrix.score_view.relative", "relative"),
    ("matrix.score_view.absolute", "absolute"),
)

EXTEND_ROOT_OBJECT_NAME = "ValidationGradientExcendRoot"
EXTEND_LANGUAGE_BUTTON_OBJECT_NAME = "extendLanguageToggleButton"
EXTEND_WIDGET_STYLESHEET = """
#ValidationGradientExcendRoot { background-color: #15191f; color: #edf3fb; }
#ValidationGradientExcendRoot QWidget { background-color: #15191f; color: #edf3fb; }
#ValidationGradientExcendRoot QGroupBox { background-color: #1a2028; border: 1px solid #304050; border-radius: 10px; margin-top: 10px; padding: 10px; font-weight: 600; }
#ValidationGradientExcendRoot QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #d7e2ef; }
#ValidationGradientExcendRoot QListWidget, #ValidationGradientExcendRoot QTabWidget::pane, #ValidationGradientExcendRoot QScrollArea, #ValidationGradientExcendRoot QMenu, #ValidationGradientExcendRoot QMenuBar, #ValidationGradientExcendRoot QSplitter::handle { background-color: #11161d; }
#ValidationGradientExcendRoot QListWidget { border: 1px solid #28384b; border-radius: 8px; outline: none; }
#ValidationGradientExcendRoot QListWidget::item { border-radius: 8px; margin: 1px 0px; padding: 1px; }
#ValidationGradientExcendRoot QListWidget::item:selected { background-color: #275fbb; color: #ffffff; }
#ValidationGradientExcendRoot QLineEdit, #ValidationGradientExcendRoot QComboBox, #ValidationGradientExcendRoot QSpinBox, #ValidationGradientExcendRoot QDoubleSpinBox { background-color: #10151c; border: 1px solid #30445a; border-radius: 8px; padding: 6px 10px; min-height: 26px; }
#ValidationGradientExcendRoot QToolButton[toolbarButton="true"] { background-color: #1e2630; border: 1px solid #314355; border-radius: 8px; padding: 4px; min-width: 28px; min-height: 28px; max-width: 28px; max-height: 28px; }
#ValidationGradientExcendRoot QToolButton[toolbarButton="true"]:hover { background-color: #283342; border-color: #46627f; }
#ValidationGradientExcendRoot QToolButton#extendLanguageToggleButton { background-color: #275fbb; border: 1px solid #3f7ee1; border-radius: 10px; padding: 6px 14px; min-width: 40px; font-weight: 700; }
#ValidationGradientExcendRoot QToolButton[folderAction="true"] { background-color: #1d2733; border: 1px solid #30445a; border-radius: 6px; padding: 0px; min-width: 22px; min-height: 22px; max-width: 22px; max-height: 22px; font-size: 11pt; font-weight: 700; color: #dfe8f2; }
#ValidationGradientExcendRoot QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #42607f; border-radius: 4px; background-color: #10151c; }
#ValidationGradientExcendRoot QCheckBox::indicator:checked { background-color: #3270d1; border-color: #4e90ff; }
#ValidationGradientExcendRoot QProgressBar { border: 1px solid #30445a; border-radius: 6px; background-color: #10151c; text-align: center; }
#ValidationGradientExcendRoot QProgressBar::chunk { background-color: #3270d1; border-radius: 5px; }
"""

# Backward-compatible lite aliases.
LITE_ROOT_OBJECT_NAME = EXTEND_ROOT_OBJECT_NAME
LITE_LANGUAGE_BUTTON_OBJECT_NAME = EXTEND_LANGUAGE_BUTTON_OBJECT_NAME
LITE_WIDGET_STYLESHEET = EXTEND_WIDGET_STYLESHEET
