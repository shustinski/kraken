"""UI builder functions for :class:`PolygonExtractionWidget`.

These functions were extracted from the widget god-class during the
production-ready refactor. They are attached back onto the widget class as
bound methods via attribute assignment (see ``contour.widget``), which
preserves all ``self.*`` attribute access without behavioral changes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from PyQt6.QtCore import QEvent, QPoint, QSize, Qt, QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStyle,
    QTableWidget,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)
from ..application.processing import (
    VIA_DISPLAY_MODE_CIRCLE,
    VIA_DISPLAY_MODE_RECTANGLE,
    VIA_SEARCH_MODE_HEURISTIC,
    VIA_SEARCH_MODE_HYBRID,
    VIA_SEARCH_MODE_TEMPLATE,
)
from ..application.vector_geometry_postprocess import VectorGeometrySettings
from ..contour_extractor import APPROXIMATION_MODE_MAP, RETRIEVAL_MODE_MAP
from ..gamification.ui import GamificationPanel
from ..graphics.tools import MIN_MANUAL_STROKE_WIDTH_PX
from ..graphics_view import BrushMode, DeleteVertexMode, EditorTool, PolygonCreateMode, PolygonEditorView
from ..vision.metal_recovery.strategy_registry import (
    IMPLEMENTED_NEW_STRATEGIES,
    ParameterSpec,
    StrategySpec,
    strategy_spec,
    visible_strategy_specs,
)
from .frame_matrix_view import FrameMatrixGraphicsView
from .frame_path_list_model import FramePathListView
from .metal_strategy_i18n import (
    choice_label,
    parameter_label,
    parameter_tooltip,
    strategy_description,
    strategy_name,
)
from .no_wheel_controls import NoWheelComboBox as QComboBox
from .no_wheel_controls import NoWheelDoubleSpinBox as QDoubleSpinBox
from .no_wheel_controls import NoWheelSpinBox as QSpinBox
from .pipeline_list import PipelineListWidget
from .status_list_delegate import attach_status_row_delegate

from .builders_parts.module_state import USE_GAMIFICATION
from .builders_parts.build_strategy_parameter_page import (
    _strategy_parameter_widget,
    _build_strategy_parameter_page,
)
from .builders_parts.sync_metal_strategy_panel import _sync_metal_strategy_panel
from .builders_parts.dock_width_combo_box import _DockWidthComboBox
from .builders_parts.build_ui import (
    _connect_line_edit_with_delay,
    build_ui,
    _apply_size_hint_geometry,
)
from .builders_parts.build_path_panel import build_path_panel
from .builders_parts.build_paths_tab import build_paths_tab
from .builders_parts.build_tabs import build_tabs
from .builders_parts.build_files_tab import build_files_tab
from .builders_parts.build_pipeline_tab import build_pipeline_tab
from .builders_parts.build_extraction_tab import build_extraction_tab
from .builders_parts.build_display_tab import build_display_tab
from .builders_parts.build_help_tab import build_help_tab
from .builders_parts.build_visual_panel import build_visual_panel
from .builders_parts.build_editor_toolbar import build_editor_toolbar

if TYPE_CHECKING:
    pass

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    USE_GAMIFICATION,
    _strategy_parameter_widget,
    _build_strategy_parameter_page,
    _sync_metal_strategy_panel,
    _DockWidthComboBox,
    _connect_line_edit_with_delay,
    build_ui,
    _apply_size_hint_geometry,
    build_path_panel,
    build_paths_tab,
    build_tabs,
    build_files_tab,
    build_pipeline_tab,
    build_extraction_tab,
    build_display_tab,
    build_help_tab,
    build_visual_panel,
    build_editor_toolbar,
)
