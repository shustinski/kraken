from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from contour.application.processing import ContourExtractionSettings, ImageProcessingState
from contour.application.use_cases.contact_feedback import (
    fit_negative_contact,
    fit_positive_contact,
)
from contour.domain import PolygonData
from contour.graphics_view import EditorTool, PolygonEditorView
from contour.vision.via_detection import (
    HeuristicViaDetectorConfig,
    ViaDetection,
    analyze_via_at,
    analyze_vias_at,
    detect_vias_heuristic,
)
from contour.vision.via_detection.settings_bridge import heuristic_config_from_settings
from contour.widget import PolygonExtractionWidget


def _detection(
    *,
    score: float = 80.0,
    brightness: float = 120.0,
    contrast: float = 80.0,
    prominence: float = 80.0,
    compactness: float = 0.9,
    circularity: float = 0.9,
    aspect: float = 1.1,
    edge_sharpness: float = 0.8,
    line_coherence: float = 0.2,
    drift_ratio: float = 0.1,
) -> ViaDetection:
    diameter = 10.0
    return ViaDetection(
        x=20.0,
        y=20.0,
        bbox=(15, 15, 10, 10),
        score=score,
        diameter_estimate=diameter,
        contrast=contrast,
        prominence=prominence,
        compactness=compactness,
        aspect=aspect,
        polarity_hypothesis="bright",
        features={
            "center_brightness": brightness,
            "circularity": circularity,
            "edge_sharpness": edge_sharpness,
            "line_coherence": line_coherence,
            "center_drift": drift_ratio * diameter,
            "diameter": diameter,
        },
    )


def test_positive_feedback_changes_only_violated_roundness() -> None:
    settings = ContourExtractionSettings(
        heuristic_min_circularity=0.8,
        heuristic_min_center_brightness=100.0,
        heuristic_min_center_contrast=50.0,
        heuristic_min_peak_prominence=50.0,
        heuristic_min_compactness=0.8,
        bright_via_min_final_score=38.0,
        via_white_range_enabled=True,
        via_white_range_min=100,
        via_white_range_max=255,
    )

    result = fit_positive_contact(
        settings,
        _detection(circularity=0.6, brightness=120.0),
    )

    assert [(change.field, change.new_value) for change in result.changes] == [
        ("heuristic_min_circularity", pytest.approx(0.6))
    ]


def test_positive_feedback_expands_every_violated_boundary_outward() -> None:
    settings = ContourExtractionSettings(
        heuristic_min_circularity=0.8,
        heuristic_min_center_contrast=70.0,
        heuristic_max_elongation=1.5,
        via_white_range_enabled=True,
        via_white_range_min=140,
        via_white_range_max=220,
        bright_via_min_final_score=50.0,
    )

    result = fit_positive_contact(
        settings,
        _detection(
            score=49.94,
            brightness=120.4,
            contrast=65.04,
            circularity=0.6549,
            aspect=1.61,
        ),
    )
    changes = {change.field: change.new_value for change in result.changes}

    assert "bright_via_min_final_score" not in changes
    assert changes["heuristic_min_circularity"] == pytest.approx(0.654)
    assert changes["heuristic_min_center_contrast"] == pytest.approx(65.0)
    assert changes["heuristic_max_elongation"] == pytest.approx(1.61)
    assert "via_white_range_min" not in changes
    assert "via_white_range_max" not in changes


def test_negative_feedback_selects_largest_normalized_gap_and_midpoint() -> None:
    settings = ContourExtractionSettings(
        bright_via_min_final_score=20.0,
        heuristic_min_circularity=0.4,
        heuristic_min_compactness=0.4,
        heuristic_min_center_contrast=10.0,
        heuristic_min_peak_prominence=10.0,
        heuristic_min_edge_sharpness=0.0,
        heuristic_max_elongation=3.0,
        heuristic_max_line_coherence=0.95,
        heuristic_min_center_brightness=0.0,
        heuristic_max_center_drift_ratio=1.0,
        via_white_range_enabled=True,
        via_white_range_min=0,
        via_white_range_max=255,
    )
    removed = _detection(score=60.0, circularity=0.50, compactness=0.70)
    references = [
        _detection(score=75.0 + index, circularity=0.80 + index * 0.005, compactness=0.72)
        for index in range(10)
    ]

    result = fit_negative_contact(settings, removed, references)

    assert all(
        not change.field.startswith(("via_white_range_", "via_black_range_"))
        for change in result.changes
    )
    assert result.selected_feature == "circularity"
    assert len(result.changes) == 1
    assert result.changes[0].field == "heuristic_min_circularity"
    assert result.changes[0].new_value == pytest.approx(0.65)
    assert result.diagnostics["reference_count"] == 10


def test_negative_feedback_requires_two_references() -> None:
    result = fit_negative_contact(
        ContourExtractionSettings(),
        _detection(),
        [_detection(score=95.0)],
    )

    assert result.changes == ()
    assert result.reason == "insufficient_references"


def test_batch_analysis_matches_single_analysis() -> None:
    image = np.zeros((64, 64), dtype=np.uint8)
    yy, xx = np.ogrid[:64, :64]
    image[(xx - 20) ** 2 + (yy - 20) ** 2 <= 16] = 220
    image[(xx - 44) ** 2 + (yy - 44) ** 2 <= 16] = 220
    config = HeuristicViaDetectorConfig(
        diameter_mode="fixed",
        fixed_diameters=[8],
        bright_range_enabled=True,
        bright_range_min=100,
    )

    batch = analyze_vias_at(image, [(20.0, 20.0), (44.0, 44.0)], config)
    singles = [
        analyze_via_at(image, 20.0, 20.0, config),
        analyze_via_at(image, 44.0, 44.0, config),
    ]

    assert [item is not None for item in batch] == [True, True]
    assert [item.score for item in batch if item is not None] == pytest.approx(
        [item.score for item in singles if item is not None]
    )


def test_positive_feedback_makes_synthetic_contact_automatically_detectable() -> None:
    image = np.zeros((64, 64), dtype=np.uint8)
    yy, xx = np.ogrid[:64, :64]
    image[(xx - 32) ** 2 + (yy - 32) ** 2 <= 16] = 220
    settings = ContourExtractionSettings(
        algorithm_backend="sem",
        recognition_mode="via",
        object_type="via",
        output_mode="box",
        via_search_mode="heuristic",
        via_size_mode="fixed",
        bright_via_diameter_min=8,
        bright_via_diameter_max=8,
        via_white_range_enabled=True,
        via_white_range_min=100,
        via_white_range_max=255,
        bright_via_min_final_score=0.0,
        heuristic_min_center_contrast=1.0,
        heuristic_min_peak_prominence=1.0,
        heuristic_min_compactness=0.99,
        heuristic_min_circularity=0.0,
        heuristic_max_elongation=20.0,
        heuristic_max_center_drift_ratio=1.5,
        heuristic_max_line_coherence=1.0,
        heuristic_min_edge_sharpness=0.0,
    )
    restrictive = heuristic_config_from_settings(settings)
    assert detect_vias_heuristic(image, restrictive).accepted == []

    measured = analyze_via_at(image, 32.0, 32.0, restrictive)
    adjustment = fit_positive_contact(settings, measured)
    for change in adjustment.changes:
        setattr(settings, change.field, change.new_value)

    accepted = detect_vias_heuristic(image, heuristic_config_from_settings(settings)).accepted
    assert any(abs(item.x - 32.0) <= 1.0 and abs(item.y - 32.0) <= 1.0 for item in accepted)


def _qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _via_polygon(*, recognized: bool) -> PolygonData:
    return PolygonData(
        id=1,
        points=[(15, 15), (25, 15), (25, 25), (15, 25)],
        category="via",
        shape_hint="box",
        bbox=(15, 15, 10, 10),
        recognition_score=90.0 if recognized else None,
    )


def test_editor_emits_feedback_signals_for_add_and_recognized_delete() -> None:
    app = _qapp()
    view = PolygonEditorView()
    view.set_image(np.zeros((80, 80), dtype=np.uint8))
    view.resize(240, 240)
    view.show()
    app.processEvents()
    added: list[tuple[float, float]] = []
    deleted: list[list[PolygonData]] = []
    view.manualViaAdded.connect(lambda x, y: added.append((x, y)))
    view.recognizedViasDeleted.connect(deleted.append)
    view.set_contact_recognition_mode(True)
    view.set_tool(EditorTool.ADD_VIA)

    QTest.mouseClick(
        view.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        view.mapFromScene(QPointF(40.0, 40.0)),
    )
    app.processEvents()
    assert added == [pytest.approx((40.0, 40.0))]

    view.set_polygons([_via_polygon(recognized=True)])
    view.set_tool(EditorTool.ADD_VIA)
    QTest.mouseClick(
        view.viewport(),
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
        view.mapFromScene(QPointF(20.0, 20.0)),
    )
    app.processEvents()
    assert len(deleted) == 1
    assert deleted[0][0].recognition_score == 90.0
    view.close()


def test_editor_does_not_emit_negative_feedback_for_manual_delete() -> None:
    app = _qapp()
    view = PolygonEditorView()
    view.set_image(np.zeros((80, 80), dtype=np.uint8))
    view.set_polygons([_via_polygon(recognized=False)])
    view.resize(240, 240)
    view.show()
    app.processEvents()
    deleted: list[list[PolygonData]] = []
    view.recognizedViasDeleted.connect(deleted.append)
    view.set_tool(EditorTool.ADD_VIA)

    QTest.mouseClick(
        view.viewport(),
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
        view.mapFromScene(QPointF(20.0, 20.0)),
    )
    app.processEvents()
    assert deleted == []
    view.close()


def test_delete_key_emits_negative_feedback_for_selected_recognized_via() -> None:
    app = _qapp()
    view = PolygonEditorView()
    view.set_image(np.zeros((80, 80), dtype=np.uint8))
    view.set_polygons([_via_polygon(recognized=True)])
    view.resize(240, 240)
    view.show()
    app.processEvents()
    deleted: list[list[PolygonData]] = []
    view.recognizedViasDeleted.connect(deleted.append)
    view._editor_scene.select_polygon(1)

    QTest.keyClick(view, Qt.Key.Key_Delete)
    app.processEvents()

    assert len(deleted) == 1
    assert deleted[0][0].id == 1
    view.close()


def test_widget_applies_positive_feedback_once_and_updates_visible_control() -> None:
    app = _qapp()
    widget = PolygonExtractionWidget()
    widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("via"))
    widget.via_search_mode_combo.setCurrentIndex(widget.via_search_mode_combo.findData("heuristic"))
    widget.heuristic_min_circularity_spin.setValue(0.8)
    widget.heuristic_min_center_brightness_spin.setValue(100.0)
    widget.via_white_range_min_spin.setValue(140)
    widget._workspace._current_image_path = "feedback.png"
    widget._workspace._current_state = ImageProcessingState(
        image_path="feedback.png",
        source_image=np.zeros((64, 64), dtype=np.uint8),
        preprocessed_image=np.zeros((64, 64), dtype=np.uint8),
    )
    widget.process_current_image = MagicMock()
    widget._abort_in_flight_interactive_processing = MagicMock()

    with patch(
        "contour.vision.via_detection.analyze_vias_at",
        return_value=[_detection(circularity=0.6, brightness=120.0)],
    ):
        widget._on_manual_via_added(20.0, 20.0)

    assert widget.heuristic_min_circularity_spin.value() == pytest.approx(0.6)
    assert widget.heuristic_min_center_brightness_spin.value() == pytest.approx(100.0)
    assert widget.via_white_range_min_spin.value() == 140
    assert widget.bright_via_advanced_outer.isChecked()
    assert "#FACC15" in widget.heuristic_min_circularity_spin.styleSheet()
    widget.process_current_image.assert_called_once_with(debounced=False)
    widget.close()
    app.processEvents()


def test_widget_restarts_after_positive_feedback_when_thresholds_do_not_change() -> None:
    app = _qapp()
    widget = PolygonExtractionWidget()
    widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("via"))
    widget.via_search_mode_combo.setCurrentIndex(widget.via_search_mode_combo.findData("heuristic"))
    widget.via_white_range_min_spin.setValue(100)
    widget._workspace._current_image_path = "feedback.png"
    widget._workspace._current_state = ImageProcessingState(
        image_path="feedback.png",
        source_image=np.zeros((64, 64), dtype=np.uint8),
        preprocessed_image=np.zeros((64, 64), dtype=np.uint8),
    )
    widget.process_current_image = MagicMock()
    widget._abort_in_flight_interactive_processing = MagicMock()

    with patch(
        "contour.vision.via_detection.analyze_vias_at",
        return_value=[_detection(brightness=180.0)],
    ):
        widget._on_manual_via_added(20.0, 20.0)

    widget.process_current_image.assert_called_once_with(debounced=False)
    widget._abort_in_flight_interactive_processing.assert_called_once_with(
        preview=True,
        prepared=False,
    )
    widget.close()
    app.processEvents()


def test_contact_parameter_dialog_is_available_without_debug_candidate_overlay() -> None:
    app = _qapp()
    widget = PolygonExtractionWidget()
    widget._ui_language = "en"
    widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("via"))
    widget.via_search_mode_combo.setCurrentIndex(widget.via_search_mode_combo.findData("heuristic"))
    widget.via_output_diameter_spin.setValue(8)
    widget.via_white_range_min_spin.setValue(100)
    widget.debug_candidates_checkbox.setChecked(False)
    image = np.zeros((64, 64), dtype=np.uint8)
    yy, xx = np.ogrid[:64, :64]
    image[(xx - 32) ** 2 + (yy - 32) ** 2 <= 16] = 220
    polygon = _via_polygon(recognized=True)
    polygon.bbox = (28, 28, 8, 8)
    polygon.points = [(28, 28), (36, 28), (36, 36), (28, 36)]
    widget._workspace._current_image_path = "feedback.png"
    widget._workspace._current_state = ImageProcessingState(
        image_path="feedback.png",
        source_image=image,
        preprocessed_image=image,
        polygons=[polygon],
        debug_candidates=[],
    )
    widget._show_nonblocking_via_debug_message = MagicMock()

    assert widget._via_debug_inspection_enabled()
    widget._on_via_debug_requested(polygon)

    widget._show_nonblocking_via_debug_message.assert_called_once()
    message = widget._show_nonblocking_via_debug_message.call_args.args[1]
    assert "Measured contact parameters:" in message
    assert "Center brightness:" in message
    assert "Final score:" in message
    widget.close()
    app.processEvents()


def test_widget_applies_negative_feedback_from_ten_best_references() -> None:
    app = _qapp()
    widget = PolygonExtractionWidget()
    widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("via"))
    widget.via_search_mode_combo.setCurrentIndex(widget.via_search_mode_combo.findData("heuristic"))
    widget.heuristic_min_circularity_spin.setValue(0.4)
    widget.bright_via_min_final_score_spin.setValue(20.0)
    references = []
    for index in range(12):
        polygon = _via_polygon(recognized=True)
        polygon.id = index + 2
        polygon.bbox = (5 + index * 4, 30, 3, 3)
        polygon.recognition_score = 100.0 - index
        references.append(polygon)
    widget._workspace._current_image_path = "feedback.png"
    widget._workspace._current_state = ImageProcessingState(
        image_path="feedback.png",
        source_image=np.zeros((80, 80), dtype=np.uint8),
        preprocessed_image=np.zeros((80, 80), dtype=np.uint8),
        polygons=references,
    )
    widget.process_current_image = MagicMock()
    widget._abort_in_flight_interactive_processing = MagicMock()
    removed = _via_polygon(recognized=True)
    measured = [
        _detection(score=60.0, circularity=0.5),
        *[
            _detection(score=75.0 + index, circularity=0.8 + index * 0.005)
            for index in range(10)
        ],
    ]

    with patch("contour.vision.via_detection.analyze_vias_at", return_value=measured):
        widget._on_recognized_vias_deleted([removed])

    assert widget.heuristic_min_circularity_spin.value() == pytest.approx(0.65)
    widget.process_current_image.assert_called_once_with(debounced=False)
    widget.close()
    app.processEvents()
