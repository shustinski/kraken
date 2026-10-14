"""build_extraction_tab: build extraction tab."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)
from ....application.processing import (
    VIA_SEARCH_MODE_HEURISTIC,
    VIA_SEARCH_MODE_HYBRID,
    VIA_SEARCH_MODE_TEMPLATE,
)
from contour.contour_extractor import APPROXIMATION_MODE_MAP, RETRIEVAL_MODE_MAP
from ....vision.metal_recovery.strategy_registry import (
    IMPLEMENTED_NEW_STRATEGIES,
    strategy_spec,
    visible_strategy_specs,
)
from contour.ui.no_wheel_controls import NoWheelComboBox as QComboBox
from contour.ui.no_wheel_controls import NoWheelDoubleSpinBox as QDoubleSpinBox
from contour.ui.no_wheel_controls import NoWheelSpinBox as QSpinBox
from contour.ui.builders_parts.build_strategy_parameter_page import _build_strategy_parameter_page
from contour.ui.builders_parts.dock_width_combo_box import _DockWidthComboBox
from contour.ui.builders_parts.sync_metal_strategy_panel import _sync_metal_strategy_panel

from .build_extraction_tab_sections.section_1 import _build_extraction_tab_section_1
from .build_extraction_tab_sections.section_2 import _build_extraction_tab_section_2
from .build_extraction_tab_sections.section_3 import _build_extraction_tab_section_3
from .build_extraction_tab_sections.section_4 import _build_extraction_tab_section_4
from .build_extraction_tab_sections.section_5 import _build_extraction_tab_section_5

def build_extraction_tab(self) -> QWidget:
    contour_layout, layout, tab = _build_extraction_tab_section_1(self)
    _expert_widget, _label = _build_extraction_tab_section_2(self)
    _cond_l, _metal_basic_form, _rm_row = _build_extraction_tab_section_3(self, _expert_widget, _label)
    _build_extraction_tab_section_4(self, _metal_basic_form)
    _build_extraction_tab_section_5(self, _cond_l, _rm_row, contour_layout, layout, tab)
    return tab
