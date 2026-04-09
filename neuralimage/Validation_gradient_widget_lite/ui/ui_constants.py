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
SETTINGS_ERROR_VIEW_KEY = "ui/error_view_settings"
SETTINGS_DETAILS_VIEW_KEY = "ui/details_view_settings"
SETTINGS_LANGUAGE_KEY = "ui/language"
SETTINGS_ORIGINAL_FOLDER_KEY = "ui/original_folder"
SETTINGS_GT_FOLDER_KEY = "ui/gt_folder"

FOLDER_CHECKED_ROLE = int(Qt.ItemDataRole.UserRole) + 1
FOLDER_LABEL_ROLE = int(Qt.ItemDataRole.UserRole) + 2

DEFAULT_COMPARISON_MODE = ComparisonMode.DISAGREEMENT
DEFAULT_CELL_SIZE = 15
DEFAULT_GRADIENT_NAME = "viridis"
DEFAULT_ERROR_WINDOW = (0.10, 0.90)
DEFAULT_MATRIX_METRIC_KEY = "overall_frame_score"
DEFAULT_ANALYSIS_MODE = INTER_MODEL_ANALYSIS_MODE
DEFAULT_METRIC_SCOPE = ""
DEFAULT_MATRIX_LAYOUT_MODE = "indexed_grid"
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
DEFAULT_CONFIDENCE_UNCERTAINTY_DELTA = 0.10
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
THUMBNAIL_SIZE_RANGE = (2, 128)
BOUNDARY_RADIUS_RANGE = (1, 16)
CONFIDENCE_UNCERTAINTY_DELTA_RANGE = (0.01, 0.49)
POINT_MATCH_RADIUS_RANGE = (1.0, 64.0)
POINT_CONFIDENCE_RADIUS_RANGE = (1, 16)
MASK_THRESHOLD_RANGE = (0.0, 1.0)
EXPORT_TOP_K_RANGE = (1, 10000)
EXPORT_PERCENT_RANGE = (1, 100)
EXPORT_PERCENTILE_RANGE = (1, 100)
EXPORT_NEIGHBOR_RANGE = (0, 16)
CONTROL_PANEL_SPLITTER_SIZES = (400, 1100)
SETTINGS_LABEL_MIN_WIDTH = 148
METRIC_SETTINGS_LABEL_MIN_WIDTH = 220
METRIC_SETTINGS_WIDGET_MIN_WIDTH = 520
METRIC_SETTINGS_COMBO_MIN_CONTENTS_LENGTH = 24
OVERVIEW_PANEL_MAX_WIDTH = 300
FOLDER_BUTTON_SIZE = 20
FOLDER_ROW_MIN_HEIGHT = 28
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
    "viridis": ((0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)), (0.75, (94, 201, 98)), (1.0, (253, 231, 37))),
    "inferno": ((0.0, (0, 0, 4)), (0.25, (87, 15, 109)), (0.5, (187, 55, 84)), (0.75, (249, 142, 8)), (1.0, (252, 255, 164))),
    "plasma": ((0.0, (13, 8, 135)), (0.25, (126, 3, 168)), (0.5, (203, 71, 119)), (0.75, (248, 149, 64)), (1.0, (240, 249, 33))),
    "cividis": ((0.0, (0, 32, 76)), (0.25, (40, 71, 112)), (0.5, (87, 109, 115)), (0.75, (140, 145, 110)), (1.0, (253, 233, 69))),
    "magma": ((0.0, (0, 0, 4)), (0.25, (80, 18, 123)), (0.5, (182, 54, 121)), (0.75, (251, 136, 97)), (1.0, (252, 253, 191))),
    "turbo": ((0.0, (48, 18, 59)), (0.25, (50, 101, 220)), (0.5, (35, 182, 121)), (0.75, (245, 209, 66)), (1.0, (122, 4, 3))),
    "gray": ((0.0, (0, 0, 0)), (1.0, (255, 255, 255))),
    "hot": ((0.0, (10, 0, 0)), (0.33, (180, 0, 0)), (0.66, (255, 180, 0)), (1.0, (255, 255, 255))),
    "coolwarm": ((0.0, (59, 76, 192)), (0.5, (221, 221, 221)), (1.0, (180, 4, 38))),
}
GRADIENT_LABELS = {name: name.title().replace("_", "-") for name in GRADIENT_PRESETS}

GEOMETRY_MODE_OPTIONS = (
    (GeometryMode.MASK.label, GeometryMode.MASK.value),
    (GeometryMode.POINT.label, GeometryMode.POINT.value),
    (GeometryMode.AUTO.label, GeometryMode.AUTO.value),
)

POLYGON_CONFIDENCE_SUMMARY_OPTIONS = (
    ('polygon_confidence_summary.weighted', 'weighted'),
    ('polygon_confidence_summary.core', 'core'),
)

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
