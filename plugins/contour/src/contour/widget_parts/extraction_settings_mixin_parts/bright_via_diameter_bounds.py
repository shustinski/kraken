"""extraction_settings_mixin: bright via diameter bounds."""
from __future__ import annotations


class BrightViaDiameterBoundsMixin:
    def _bright_via_diameter_bounds_from_widgets(self) -> tuple[int, int]:
        first = int(self.bright_via_diameter_min_spin.value())
        second = int(self.bright_via_diameter_max_spin.value())
        return min(first, second), max(first, second)
