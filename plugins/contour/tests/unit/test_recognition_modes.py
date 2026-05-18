from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication

from contour.application.processing import (
    ContourExtractionSettings,
    normalize_metal_segmentation_method,
    normalize_metal_sensitivity,
    normalize_via_search_mode,
)
from contour.ui.via_presets import built_in_via_presets
from contour.vision.metal_recovery.detector import (
    _normalize_metal_extraction_mode,
    _normalize_metal_segmentation_method,
    _normalize_metal_sensitivity_token,
)
from contour.widget import PolygonExtractionWidget


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("heuristic", "heuristic"),
        ("template", "template"),
        ("bright_tophat_dog", "bright_tophat_dog"),
        ("hybrid", "heuristic"),
        ("blob", "heuristic"),
        ("unknown", "heuristic"),
        (None, "heuristic"),
    ],
)
def test_normalize_via_search_mode(raw: object, expected: str) -> None:
    assert normalize_via_search_mode(raw) == expected
    assert ContourExtractionSettings.from_dict({"via_search_mode": raw}).via_search_mode == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("без", "none"),
        ("без сегментации", "none"),
        ("гибрид", "hybrid"),
        ("гибридная", "hybrid"),
        ("адаптивная", "adaptive"),
        ("otsu", "otsu"),
    ],
)
def test_russian_metal_segmentation_values_normalize(raw: str, expected: str) -> None:
    assert normalize_metal_segmentation_method(raw) == expected
    assert _normalize_metal_extraction_mode(raw) == expected


def test_russian_metal_threshold_and_sensitivity_values_normalize() -> None:
    assert _normalize_metal_segmentation_method("адаптивная") == "adaptive"
    assert normalize_metal_sensitivity("низкая") == "low"
    assert normalize_metal_sensitivity("средняя") == "medium"
    assert normalize_metal_sensitivity("высокая") == "high"
    assert _normalize_metal_sensitivity_token("низкая") == "low"
    assert _normalize_metal_sensitivity_token("средняя") == "medium"
    assert _normalize_metal_sensitivity_token("высокая") == "high"


def test_mode_switching_hides_irrelevant_ui_settings() -> None:
    app = QApplication.instance() or QApplication([])
    widget = PolygonExtractionWidget()
    try:
        widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("disabled"))
        app.processEvents()
        assert not widget.recognition_stack.isHidden()
        assert widget.bright_via_group.isHidden()

        widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("conductors"))
        app.processEvents()
        assert not widget.metal_basic_group.isHidden()
        assert widget.bright_via_group.isHidden()

        widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("via"))
        app.processEvents()
        assert not widget.bright_via_group.isHidden()
        assert widget.recognition_stack.isHidden()
        assert [widget.via_search_mode_combo.itemData(i) for i in range(widget.via_search_mode_combo.count())] == [
            "template",
            "bright_tophat_dog",
        ]
    finally:
        widget.close()
        widget.deleteLater()


def test_conductors_mode_stays_enabled_after_disabled_toggle() -> None:
    app = QApplication.instance() or QApplication([])
    widget = PolygonExtractionWidget()
    try:
        widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("disabled"))
        app.processEvents()
        widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("conductors"))
        app.processEvents()
        assert widget.recognition_mode_combo.currentData() == "conductors"
    finally:
        widget.close()
        widget.deleteLater()


def test_via_profiles_set_expected_parameters() -> None:
    presets = built_in_via_presets("ru")
    assert set(presets) == {
        "Стандартный",
        "Малые via",
        "Крупные via",
        "Светлые via",
        "Тёмные via",
        "Via с кольцом",
        "Слабый контраст",
    }
    assert presets["Светлые via"]["via_search_mode"] == "bright_tophat_dog"
    assert presets["Малые via"]["bright_via_diameter_max"] < presets["Крупные via"]["bright_via_diameter_min"]
