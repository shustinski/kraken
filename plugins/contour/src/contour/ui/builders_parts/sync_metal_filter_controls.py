"""builders: enable recognized-metal filter fields from their checkboxes."""
from __future__ import annotations

from typing import TYPE_CHECKING

from contour.application import metal_filters as mf

if TYPE_CHECKING:
    from contour.widget import PolygonExtractionWidget

# The rim checks only relax the object/background contrast filter.
_CONTRAST_EXEMPTIONS = (mf.FILTER_MIN_OBJECT_RIM_CONTRAST, mf.FILTER_MIN_OBJECT_RIM_AREA_FRACTION)


def enabled_metal_filters(self: PolygonExtractionWidget) -> tuple[str, ...]:
    checkboxes = getattr(self, "metal_filter_checkboxes", None)
    if not checkboxes:
        return mf.DEFAULT_ENABLED_METAL_FILTERS
    return tuple(key for key in mf.METAL_FILTER_KEYS if key in checkboxes and checkboxes[key].isChecked())


def set_enabled_metal_filters(self: PolygonExtractionWidget, enabled: object) -> None:
    checkboxes = getattr(self, "metal_filter_checkboxes", None)
    if not checkboxes:
        return
    keys = set(mf.normalize_enabled_metal_filters(enabled))
    for key, checkbox in checkboxes.items():
        previous = checkbox.blockSignals(True)
        checkbox.setChecked(key in keys)
        checkbox.blockSignals(previous)
    _sync_metal_filter_controls(self)


def _sync_metal_filter_controls(self: PolygonExtractionWidget) -> None:
    checkboxes = getattr(self, "metal_filter_checkboxes", None)
    fields = getattr(self, "metal_filter_fields", None)
    if not checkboxes or not fields:
        return
    contrast_on = checkboxes[mf.FILTER_MIN_OBJECT_SOURCE_CONTRAST].isChecked()
    for key, checkbox in checkboxes.items():
        available = contrast_on or key not in _CONTRAST_EXEMPTIONS
        checkbox.setEnabled(available)
        fields[key].setEnabled(available and checkbox.isChecked())
