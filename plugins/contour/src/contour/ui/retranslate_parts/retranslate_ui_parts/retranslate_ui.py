"""retranslate_ui: retranslate ui."""
from __future__ import annotations
from typing import TYPE_CHECKING
from PyQt6.QtCore import QSignalBlocker
from ....vision.metal_recovery.strategy_registry import strategy_spec
from contour.ui.bright_via_i18n import retranslate_bright_via_panel
from contour.ui.i18n_content import PIPELINE_CONTROL_TOOLTIPS, _localized_text
from contour.ui.metal_debug_i18n import retranslate_metal_debug_visual_combo
from contour.ui.metal_strategy_i18n import strategy_name
from contour.ui.retranslate_parts.retranslate_strategy_parameter_pages import (
    _retranslate_strategy_parameter_pages,
    _set_form_label,
)

from .retranslate_ui_sections.section_1 import _retranslate_ui_section_1
from .retranslate_ui_sections.section_2 import _retranslate_ui_section_2
from .retranslate_ui_sections.section_3 import _retranslate_ui_section_3
from .retranslate_ui_sections.section_4 import _retranslate_ui_section_4

def retranslate_ui(self: PolygonExtractionWidget) -> None:
    if not hasattr(self, "control_tabs"):
        return
    index, selected_operation, selected_pipeline_row, tooltip, tooltip_key, widget = _retranslate_ui_section_1(self)
    _retranslate_ui_section_2(self, index)
    tooltip_key, widget = _retranslate_ui_section_3(self, tooltip_key, widget)
    _retranslate_ui_section_4(self, selected_operation, selected_pipeline_row, tooltip, tooltip_key, widget)

if TYPE_CHECKING:
    from contour.widget import PolygonExtractionWidget
