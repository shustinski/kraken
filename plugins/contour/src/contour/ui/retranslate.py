"""Retranslate implementation for :class:`PolygonExtractionWidget`.

Split out of ``contour.widget`` during the production-ready refactor.
The function is bound as a method of the widget via attribute assignment, so
``self`` refers to the original widget and all attribute access is unchanged.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QComboBox, QFormLayout, QGroupBox
from ..vision.metal_recovery.strategy_registry import strategy_spec
from .bright_via_i18n import retranslate_bright_via_panel
from .i18n_content import PIPELINE_CONTROL_TOOLTIPS, _localized_text
from .metal_debug_i18n import retranslate_metal_debug_visual_combo
from .metal_strategy_i18n import choice_label, parameter_label, parameter_tooltip, strategy_name

from .retranslate_parts.retranslate_strategy_parameter_pages import (
    _set_form_label,
    _retranslate_strategy_parameter_pages,
)
from .retranslate_parts.retranslate_ui import retranslate_ui

if TYPE_CHECKING:
    from contour.widget import PolygonExtractionWidget

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _set_form_label,
    _retranslate_strategy_parameter_pages,
    retranslate_ui,
)
