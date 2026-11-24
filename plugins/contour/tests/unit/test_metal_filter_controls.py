from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from contour.application.metal_filters import DEFAULT_ENABLED_METAL_FILTERS, METAL_FILTER_KEYS
from contour.application.processing import ContourExtractionSettings
from contour.widget import PolygonExtractionWidget


def _qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _widget_with_advanced_open() -> PolygonExtractionWidget:
    _qapp()
    widget = PolygonExtractionWidget()
    widget.metal_advanced_group.setChecked(True)
    return widget


def _filter_form_fields(widget: PolygonExtractionWidget) -> list[object]:
    form = widget.metal_filter_group.layout()
    return [form.itemAt(row, form.ItemRole.FieldRole).widget() for row in range(form.rowCount())]


def test_width_lives_in_filter_group_with_a_checkbox_per_filter() -> None:
    _qapp()
    widget = PolygonExtractionWidget()

    fields = _filter_form_fields(widget)
    assert widget.metal_min_width_spin in fields
    assert widget.metal_max_width_spin in fields
    assert widget.metal_basic_group.layout().labelForField(widget.metal_min_width_spin) is None
    assert set(widget.metal_filter_checkboxes) == set(METAL_FILTER_KEYS)
    for key, checkbox in widget.metal_filter_checkboxes.items():
        assert widget.metal_filter_group.layout().labelForField(widget.metal_filter_fields[key]) is checkbox
    widget.close()


def test_defaults_enable_only_area_and_point_count() -> None:
    widget = _widget_with_advanced_open()

    checked = {key for key, box in widget.metal_filter_checkboxes.items() if box.isChecked()}

    assert checked == set(DEFAULT_ENABLED_METAL_FILTERS)
    assert widget._current_contour_settings().metal_enabled_filters == DEFAULT_ENABLED_METAL_FILTERS
    assert widget.metal_min_area_spin.isEnabled()
    assert not widget.metal_min_width_spin.isEnabled()
    widget.close()


def test_filter_checkboxes_follow_applied_settings() -> None:
    widget = _widget_with_advanced_open()
    enabled = ("min_object_source_contrast", "min_object_rim_contrast", "min_trace_width")

    widget._set_extraction_settings(ContourExtractionSettings(metal_enabled_filters=enabled))

    assert widget._current_contour_settings().metal_enabled_filters == enabled
    assert widget.metal_min_width_spin.isEnabled()
    assert not widget.metal_min_area_spin.isEnabled()
    assert widget.metal_filter_checkboxes["min_object_rim_area_fraction"].isEnabled()
    widget.close()


def test_rim_checks_are_unavailable_while_contrast_filter_is_off() -> None:
    widget = _widget_with_advanced_open()

    widget.metal_filter_checkboxes["min_object_source_contrast"].setChecked(False)

    assert not widget.metal_filter_checkboxes["min_object_rim_contrast"].isEnabled()
    assert not widget.metal_min_object_rim_contrast_spin.isEnabled()
    widget.metal_filter_checkboxes["min_object_source_contrast"].setChecked(True)
    assert widget.metal_filter_checkboxes["min_object_rim_contrast"].isEnabled()
    widget.close()
