"""Store only mismatch-only UI constants used by the lite widget."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from ..core.domain import ComparisonMode

SETTINGS_ORG = "ValidationGradientWidgetLite"
SETTINGS_APP = "ValidationGradientWidgetLite"
SETTINGS_FOLDERS_KEY = "ui/folder_manager"
SETTINGS_BUILD_KEY = "ui/build_settings"
SETTINGS_ERROR_VIEW_KEY = "ui/error_view_settings"
SETTINGS_LANGUAGE_KEY = "ui/language"

FOLDER_CHECKED_ROLE = int(Qt.ItemDataRole.UserRole) + 1
FOLDER_LABEL_ROLE = int(Qt.ItemDataRole.UserRole) + 2

DEFAULT_COMPARISON_MODE = ComparisonMode.DISAGREEMENT
DEFAULT_CELL_SIZE = 16
DEFAULT_GRADIENT_NAME = "viridis"
DEFAULT_ERROR_WINDOW = (0.10, 0.90)
DEFAULT_SCORE_VIEW_MODE = "relative"
DEFAULT_MATRIX_LAYOUT_MODE = "indexed_grid"
DEFAULT_TOTAL_FRAMES = 9999
DEFAULT_FRAMES_PER_ROW = 99
DEFAULT_MATRIX_ROWS = 100
DEFAULT_MATRIX_COLUMNS = 100
DEFAULT_WINDOW_WIDTH = 1400
DEFAULT_WINDOW_HEIGHT = 900
REQUIRED_COMPARE_FOLDER_COUNT = 2
NORMALIZATION_EPSILON = 1e-9

TOTAL_FRAMES_RANGE = (0, 10_000_000)
FRAMES_PER_ROW_RANGE = (0, 100_000)
MATRIX_ROWS_RANGE = (1, 100_000)
MATRIX_COLUMNS_RANGE = (1, 100_000)
THUMBNAIL_SIZE_RANGE = (2, 128)
CONTROL_PANEL_SPLITTER_SIZES = (320, 1080)
SETTINGS_LABEL_MIN_WIDTH = 170
OVERVIEW_PANEL_MAX_WIDTH = 220
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
    "viridis": (
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.5, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ),
    "inferno": (
        (0.0, (0, 0, 4)),
        (0.25, (87, 15, 109)),
        (0.5, (187, 55, 84)),
        (0.75, (249, 142, 8)),
        (1.0, (252, 255, 164)),
    ),
    "plasma": (
        (0.0, (13, 8, 135)),
        (0.25, (126, 3, 168)),
        (0.5, (203, 71, 119)),
        (0.75, (248, 149, 64)),
        (1.0, (240, 249, 33)),
    ),
    "cividis": (
        (0.0, (0, 32, 76)),
        (0.25, (40, 71, 112)),
        (0.5, (87, 109, 115)),
        (0.75, (140, 145, 110)),
        (1.0, (253, 233, 69)),
    ),
    "magma": (
        (0.0, (0, 0, 4)),
        (0.25, (80, 18, 123)),
        (0.5, (182, 54, 121)),
        (0.75, (251, 136, 97)),
        (1.0, (252, 253, 191)),
    ),
    "turbo": (
        (0.0, (48, 18, 59)),
        (0.25, (50, 101, 220)),
        (0.5, (35, 182, 121)),
        (0.75, (245, 209, 66)),
        (1.0, (122, 4, 3)),
    ),
    "gray": (
        (0.0, (0, 0, 0)),
        (1.0, (255, 255, 255)),
    ),
    "hot": (
        (0.0, (10, 0, 0)),
        (0.33, (180, 0, 0)),
        (0.66, (255, 180, 0)),
        (1.0, (255, 255, 255)),
    ),
    "cool": (
        (0.0, (0, 255, 255)),
        (1.0, (255, 0, 255)),
    ),
    "coolwarm": (
        (0.0, (59, 76, 192)),
        (0.5, (221, 221, 221)),
        (1.0, (180, 4, 38)),
    ),
    "spectral": (
        (0.0, (94, 79, 162)),
        (0.25, (50, 136, 189)),
        (0.5, (255, 255, 191)),
        (0.75, (252, 141, 89)),
        (1.0, (158, 1, 66)),
    ),
    "red_green": (
        (0.0, (165, 0, 38)),
        (0.5, (255, 255, 191)),
        (1.0, (0, 104, 55)),
    ),
    "blue_red": (
        (0.0, (49, 54, 149)),
        (0.5, (247, 247, 247)),
        (1.0, (165, 0, 38)),
    ),
    "cubehelix": (
        (0.0, (0, 0, 0)),
        (0.25, (40, 32, 108)),
        (0.5, (120, 84, 92)),
        (0.75, (196, 160, 88)),
        (1.0, (255, 255, 255)),
    ),
}

LITE_ROOT_OBJECT_NAME = "ValidationGradientLiteRoot"
LITE_LANGUAGE_BUTTON_OBJECT_NAME = "liteLanguageToggleButton"
LITE_WIDGET_STYLESHEET = """
#ValidationGradientLiteRoot {
    background-color: #15191f;
    color: #edf3fb;
}

#ValidationGradientLiteRoot QWidget {
    background-color: #15191f;
    color: #edf3fb;
}

#ValidationGradientLiteRoot QGroupBox {
    background-color: #1a2028;
    border: 1px solid #304050;
    border-radius: 10px;
    margin-top: 10px;
    padding: 10px;
    font-weight: 600;
}

#ValidationGradientLiteRoot QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #d7e2ef;
}

#ValidationGradientLiteRoot QListWidget,
#ValidationGradientLiteRoot QTabWidget::pane,
#ValidationGradientLiteRoot QScrollArea,
#ValidationGradientLiteRoot QMenu,
#ValidationGradientLiteRoot QMenuBar,
#ValidationGradientLiteRoot QSplitter::handle {
    background-color: #11161d;
}

#ValidationGradientLiteRoot QListWidget {
    border: 1px solid #28384b;
    border-radius: 8px;
    outline: none;
}

#ValidationGradientLiteRoot QListWidget::item {
    border-radius: 8px;
    margin: 1px 0px;
    padding: 1px;
}

#ValidationGradientLiteRoot QListWidget::item:selected {
    background-color: #275fbb;
    color: #ffffff;
}

#ValidationGradientLiteRoot QLineEdit,
#ValidationGradientLiteRoot QComboBox,
#ValidationGradientLiteRoot QSpinBox,
#ValidationGradientLiteRoot QDoubleSpinBox {
    background-color: #10151c;
    border: 1px solid #30445a;
    border-radius: 8px;
    padding: 6px 10px;
    min-height: 26px;
}

#ValidationGradientLiteRoot QLineEdit:focus,
#ValidationGradientLiteRoot QComboBox:focus,
#ValidationGradientLiteRoot QSpinBox:focus,
#ValidationGradientLiteRoot QDoubleSpinBox:focus {
    border: 1px solid #5aa0ff;
}

#ValidationGradientLiteRoot QToolButton[liteToolbarButton="true"] {
    background-color: #1e2630;
    border: 1px solid #314355;
    border-radius: 8px;
    padding: 4px;
    min-width: 28px;
    min-height: 28px;
    max-width: 28px;
    max-height: 28px;
}

#ValidationGradientLiteRoot QToolButton[liteToolbarButton="true"]:hover {
    background-color: #283342;
    border-color: #46627f;
}

#ValidationGradientLiteRoot QToolButton[liteToolbarButton="true"]:pressed {
    background-color: #172029;
}

#ValidationGradientLiteRoot QToolButton#liteLanguageToggleButton {
    background-color: #275fbb;
    border: 1px solid #3f7ee1;
    border-radius: 10px;
    padding: 6px 14px;
    min-width: 40px;
    font-weight: 700;
}

#ValidationGradientLiteRoot QToolButton#liteLanguageToggleButton:hover {
    background-color: #3270d1;
}

#ValidationGradientLiteRoot QToolButton[folderAction="true"] {
    background-color: #1d2733;
    border: 1px solid #30445a;
    border-radius: 6px;
    padding: 0px;
    min-width: 22px;
    min-height: 22px;
    max-width: 22px;
    max-height: 22px;
    font-size: 11pt;
    font-weight: 700;
    color: #dfe8f2;
}

#ValidationGradientLiteRoot QToolButton[folderAction="true"]:hover {
    background-color: #243240;
    border-color: #46627f;
}

#ValidationGradientLiteRoot QToolButton[folderAction="true"]:disabled {
    background-color: #151b22;
    border-color: #273341;
    color: #6e7d8d;
}

#ValidationGradientLiteRoot QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #42607f;
    border-radius: 4px;
    background-color: #10151c;
}

#ValidationGradientLiteRoot QCheckBox::indicator:checked {
    background-color: #3270d1;
    border-color: #4e90ff;
}

#ValidationGradientLiteRoot QProgressBar {
    border: 1px solid #30445a;
    border-radius: 6px;
    background-color: #10151c;
    text-align: center;
}

#ValidationGradientLiteRoot QProgressBar::chunk {
    background-color: #3270d1;
    border-radius: 5px;
}

#ValidationGradientLiteRoot QMenuBar {
    border-bottom: 1px solid #223244;
    padding: 4px;
}

#ValidationGradientLiteRoot QMenuBar::item {
    background: transparent;
    padding: 5px 10px;
    margin: 1px 2px;
    border-radius: 6px;
}

#ValidationGradientLiteRoot QMenuBar::item:selected {
    background-color: #203142;
}

#ValidationGradientLiteRoot QMenu {
    border: 1px solid #28384b;
    border-radius: 8px;
    padding: 6px;
}

#ValidationGradientLiteRoot QMenu::item {
    padding: 6px 14px;
    border-radius: 6px;
}

#ValidationGradientLiteRoot QMenu::item:selected {
    background-color: #2f6fd5;
}
"""

GRADIENT_LABELS = {
    "viridis": "Viridis",
    "inferno": "Inferno",
    "plasma": "Plasma",
    "cividis": "Cividis",
    "magma": "Magma",
    "turbo": "Turbo",
    "gray": "Gray",
    "hot": "Hot",
    "cool": "Cool",
    "coolwarm": "Coolwarm",
    "spectral": "Spectral",
    "red_green": "Red-Green",
    "blue_red": "Blue-Red",
    "cubehelix": "Cubehelix",
}
