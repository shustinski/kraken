"""extraction_controls_mixin: apply bright via tooltips."""
from __future__ import annotations


from .apply_bright_via_tooltips_sections.section_1 import _apply_bright_via_tooltips_section_1
from .apply_bright_via_tooltips_sections.section_2 import _apply_bright_via_tooltips_section_2
from .apply_bright_via_tooltips_sections.section_3 import _apply_bright_via_tooltips_section_3
from .apply_bright_via_tooltips_sections.section_4 import _apply_bright_via_tooltips_section_4

class ApplyBrightViaTooltipsMixin:
    def _apply_bright_via_tooltips(self) -> None:
        if not hasattr(self, "bright_via_diameter_min_spin"):
            return
        tt = _apply_bright_via_tooltips_section_1(self)
        label_widget = _apply_bright_via_tooltips_section_2(self, tt)
        _apply_bright_via_tooltips_section_3(self, label_widget, tt)
        _apply_bright_via_tooltips_section_4(self, tt)
