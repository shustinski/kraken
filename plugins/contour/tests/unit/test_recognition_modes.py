from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication

from contour.application.processing import (
    ContourExtractionSettings,
    normalize_metal_segmentation_strategy,
    normalize_via_search_mode,
)
from contour.ui.via_presets import built_in_via_presets
from contour.vision.metal_recovery.detector import _normalize_metal_extraction_mode
from contour.vision.metal_recovery.segmentation import migrate_legacy_metal_settings
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
        ("", "heuristic"),
    ],
)
def test_normalize_via_search_mode(raw: object, expected: str) -> None:
    assert normalize_via_search_mode(raw) == expected
    assert ContourExtractionSettings.from_dict({"via_search_mode": raw}).via_search_mode == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("без", "edges"),
        ("без сегментации", "edges"),
        ("гибрид", "auto"),
        ("гибридная", "auto"),
        ("адаптивная", "local_adaptive"),
        ("otsu", "legacy_otsu"),
        ("auto", "auto"),
        ("sauvola", "sauvola"),
    ],
)
def test_russian_metal_segmentation_values_normalize(raw: str, expected: str) -> None:
    assert normalize_metal_segmentation_strategy(raw) == expected
    assert _normalize_metal_extraction_mode(raw) == expected


def test_legacy_metal_settings_migration() -> None:
    migrated = migrate_legacy_metal_settings(
        {"metal_sensitivity_0_100": 77, "metal_sensitivity": "low", "metal_segmentation_method": "otsu"}
    )
    assert "metal_contrast_bias" in migrated
    assert migrated["metal_segmentation_strategy"] == "auto"
    settings = ContourExtractionSettings.from_dict(migrated)
    assert settings.metal_segmentation_strategy == "auto"


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
            "heuristic",
            "bright_tophat_dog",
            "template",
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
    assert presets["Стандартный"]["via_size_mode"] == "fixed"
    assert presets["Стандартный"]["bright_via_diameter_min"] == presets["Стандартный"]["bright_via_diameter_max"]
    assert presets["Малые via"]["bright_via_diameter_max"] < presets["Крупные via"]["bright_via_diameter_min"]
