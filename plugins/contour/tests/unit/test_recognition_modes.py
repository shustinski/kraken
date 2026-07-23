from __future__ import annotations

import json

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QHeaderView,
    QSpinBox,
)

from contour.application.processing import (
    ContourDebugCandidate,
    ContourExtractionSettings,
    ImageProcessingState,
    normalize_metal_segmentation_strategy,
    normalize_via_search_mode,
)
from contour.domain import PolygonData
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
        ("hybrid", "hybrid"),
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
            "template",
            "hybrid",
        ]
    finally:
        widget.close()


def test_contact_layout_stays_within_dock_and_hides_irrelevant_diameter_range() -> None:
    app = QApplication.instance() or QApplication([])
    widget = PolygonExtractionWidget()
    try:
        widget.resize(500, 800)
        widget.show()
        widget.control_tabs.setCurrentWidget(widget.extraction_tab)
        widget.main_splitter.setSizes([420, 650, 220])
        widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("via"))
        widget.via_search_mode_combo.setCurrentIndex(widget.via_search_mode_combo.findData("heuristic"))
        app.processEvents()
        assert widget.bright_via_mode_stack.isHidden()

        widget.via_search_mode_combo.setCurrentIndex(widget.via_search_mode_combo.findData("hybrid"))
        widget.via_diameter_size_mode_combo.setCurrentIndex(
            widget.via_diameter_size_mode_combo.findData("fixed")
        )
        widget._via_template_images = [np.full((8, 8), 180, dtype=np.uint8)]
        widget._refresh_via_template_list()
        app.processEvents()

        assert widget.bright_via_diameter_range_widget.isHidden()
        assert widget.bright_via_diameter_range_label_widget.isHidden()
        assert not widget.bright_via_diameter_fixed_spin.isHidden()
        header = widget.via_template_table.horizontalHeader()
        assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.Stretch
        assert header.sectionResizeMode(3) == QHeaderView.ResizeMode.Stretch
        assert sum(widget.via_template_table.columnWidth(i) for i in range(5)) <= (
            widget.via_template_table.viewport().width() + 2
        )

        widget.recognition_mode_combo.showPopup()
        app.processEvents()
        QTest.qWait(80)
        app.processEvents()
        visible_combo_rect = widget.recognition_mode_combo.visibleRegion().boundingRect()
        assert widget.recognition_mode_combo.view().window().width() == visible_combo_rect.width()
        assert widget.recognition_mode_combo.view().window().x() == (
            widget.recognition_mode_combo.mapToGlobal(QPoint(visible_combo_rect.left(), 0)).x()
        )
        widget.recognition_mode_combo.hidePopup()

        widget.via_diameter_size_mode_combo.setCurrentIndex(
            widget.via_diameter_size_mode_combo.findData("range")
        )
        app.processEvents()
        assert not widget.bright_via_diameter_range_widget.isHidden()
        assert widget.bright_via_diameter_fixed_spin.isHidden()
    finally:
        widget.close()
        widget.deleteLater()


def test_heuristic_expert_group_and_tooltips() -> None:
    app = QApplication.instance() or QApplication([])
    widget = PolygonExtractionWidget()
    try:
        widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("via"))
        widget.via_search_mode_combo.setCurrentIndex(widget.via_search_mode_combo.findData("heuristic"))
        app.processEvents()
        assert widget.bright_via_advanced_outer.title() == "Экспертные параметры"
        assert widget.bright_via_advanced_outer.isCheckable()
        assert not widget.bright_via_advanced_outer.isHidden()

        expert_widgets = [
            value
            for name, value in vars(widget).items()
            if name.startswith("heuristic_")
            and (name.endswith("_spin") or name.endswith("_checkbox"))
        ]
        assert expert_widgets
        for editor in expert_widgets:
            assert len(editor.toolTip()) >= 40
            label = widget.bright_via_form.labelForField(editor)
            if label is not None:
                assert label.toolTip() == editor.toolTip()

        widget.resize(1280, 900)
        widget.show()
        widget.bright_via_advanced_outer.setChecked(True)
        app.processEvents()
        expert_spinboxes = [
            editor
            for editor in expert_widgets
            if isinstance(editor, (QSpinBox, QDoubleSpinBox))
        ]
        assert expert_spinboxes
        assert {editor.width() for editor in expert_spinboxes} == {
            widget.heuristic_expert_spinbox_width
        }
        widget.bright_via_form.activate()
        spinbox_x_positions = {editor.geometry().x() for editor in expert_spinboxes}
        assert len(spinbox_x_positions) == 1
        assert next(iter(spinbox_x_positions)) <= widget.heuristic_expert_label_width + 20
        expert_labels = [
            widget.bright_via_form.labelForField(editor)
            for editor in expert_spinboxes
        ]
        assert {
            label.width()
            for label in expert_labels
            if label is not None
        } == {widget.heuristic_expert_label_width}
        assert widget.bright_via_form.labelForField(
            widget.heuristic_use_bilateral_checkbox
        ) is None
        assert (
            widget.heuristic_use_bilateral_checkbox.geometry().x()
            < next(iter(spinbox_x_positions))
        )

        widget.heuristic_w_contrast_spin.setValue(37.0)
        widget.heuristic_min_center_brightness_spin.setValue(123.0)
        widget.heuristic_min_circularity_spin.setValue(0.55)
        payload = widget._current_via_preset_payload()
        assert payload
        assert all(key.startswith("heuristic_") for key in payload)
        assert payload["heuristic_w_contrast"] == pytest.approx(37.0)
        assert payload["heuristic_min_center_brightness"] == pytest.approx(123.0)
        assert payload["heuristic_min_circularity"] == pytest.approx(0.55)
        original_method = widget.via_search_mode_combo.currentData()
        original_polarity = widget.via_heuristic_polarity_combo.currentData()
        original_diameter = widget.bright_via_diameter_fixed_spin.value()
        widget.heuristic_w_contrast_spin.setValue(12.0)
        widget.heuristic_min_circularity_spin.setValue(0.1)
        widget.bright_via_advanced_outer.setChecked(False)
        widget._apply_via_preset_payload(
            payload
            | {
                "via_search_mode": "template",
                "via_heuristic_polarity": "dark",
                "bright_via_diameter_min": 99,
                "bright_via_diameter_max": 99,
            }
        )
        assert widget.heuristic_w_contrast_spin.value() == pytest.approx(37.0)
        assert widget.heuristic_min_circularity_spin.value() == pytest.approx(0.55)
        assert widget.bright_via_advanced_outer.isChecked()
        assert widget.via_search_mode_combo.currentData() == original_method
        assert widget.via_heuristic_polarity_combo.currentData() == original_polarity
        assert widget.bright_via_diameter_fixed_spin.value() == original_diameter
        widget.heuristic_max_line_coherence_spin.setValue(0.41)
        widget.heuristic_use_bilateral_checkbox.setChecked(True)
        widget.bright_via_advanced_outer.setChecked(True)
        widget.heuristic_defaults_button.click()
        assert widget.heuristic_w_contrast_spin.value() == pytest.approx(25.0)
        assert widget.heuristic_min_circularity_spin.value() == pytest.approx(0.40)
        assert widget.heuristic_background_sigma_spin.value() == pytest.approx(25.0)
        assert widget.heuristic_analysis_window_scale_spin.value() == pytest.approx(3.0)
        assert widget.heuristic_min_center_brightness_spin.value() == pytest.approx(0.0)
        assert widget.heuristic_min_center_contrast_spin.value() == pytest.approx(50.0)
        assert widget.heuristic_min_peak_prominence_spin.value() == pytest.approx(50.0)
        assert widget.heuristic_local_binarize_percentile_spin.value() == pytest.approx(88.0)
        assert widget.heuristic_min_abs_peak_spin.value() == pytest.approx(0.0)
        assert widget.heuristic_seed_percentile_spin.value() == pytest.approx(90.0)
        assert not hasattr(widget, "via_search_sensitivity_combo")
        assert widget.heuristic_min_compactness_spin.value() == pytest.approx(0.9)
        assert widget.heuristic_min_compactness_spin.singleStep() == pytest.approx(0.1)
        assert widget.heuristic_max_elongation_spin.value() == pytest.approx(2.5)
        assert widget.heuristic_size_tolerance_range_spin.value() == pytest.approx(0.36)
        assert widget.heuristic_size_tolerance_fixed_spin.value() == pytest.approx(0.26)
        assert widget.heuristic_max_center_drift_ratio_spin.value() == pytest.approx(0.72)
        assert widget.heuristic_max_line_coherence_spin.value() == pytest.approx(0.82)
        assert widget.heuristic_min_edge_sharpness_spin.value() == pytest.approx(0.20)
        assert widget.heuristic_line_penalty_spin.value() == pytest.approx(3.0)
        assert widget.heuristic_border_penalty_spin.value() == pytest.approx(1.0)
        assert widget.heuristic_contrast_score_min_spin.value() == pytest.approx(3.0)
        assert widget.heuristic_contrast_score_max_spin.value() == pytest.approx(20.0)
        assert widget.heuristic_prominence_score_min_spin.value() == pytest.approx(2.0)
        assert widget.heuristic_prominence_score_max_spin.value() == pytest.approx(25.0)
        assert widget.heuristic_edge_snr_score_min_spin.value() == pytest.approx(0.70)
        assert widget.heuristic_edge_snr_score_max_spin.value() == pytest.approx(2.80)
        assert widget.heuristic_edge_quality_floor_spin.value() == pytest.approx(0.55)
        assert widget.heuristic_border_balance_scale_spin.value() == pytest.approx(2.0)
        assert not widget.heuristic_use_bilateral_checkbox.isChecked()
        assert widget.bright_via_diameter_fixed_spin.value() == original_diameter
        assert widget.heuristic_defaults_button.text() == "По умолчанию"
        assert len(widget.heuristic_defaults_button.toolTip()) >= 40
        widget.bright_via_advanced_outer.setChecked(False)
        widget.bright_via_advanced_outer.setChecked(True)
        assert widget.heuristic_w_contrast_spin.value() == 25.0
    finally:
        widget.close()
        widget.deleteLater()


def test_contact_polarity_is_in_basics_and_obsolete_debug_controls_are_removed() -> None:
    app = QApplication.instance() or QApplication([])
    widget = PolygonExtractionWidget()
    try:
        widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("via"))
        widget.via_search_mode_combo.setCurrentIndex(widget.via_search_mode_combo.findData("heuristic"))
        app.processEvents()

        assert widget.bright_via_basics_form.labelForField(widget.via_heuristic_polarity_combo) is (
            widget.bright_via_polarity_label_widget
        )
        assert widget.bright_via_polarity_label_widget.text() == "Полярность контакта"
        assert not widget.via_heuristic_polarity_combo.isHidden()
        assert widget.bright_via_display_form.labelForField(widget.via_show_detected_checkbox) is None
        assert widget.bright_via_display_form.labelForField(widget.debug_candidates_checkbox) is None
        assert widget.bright_via_display_form.labelForField(widget.bright_via_show_rejected_checkbox) is None
        assert not hasattr(widget, "preview_bright_via_mask_button")
        assert not hasattr(widget, "show_gradient_debug_button")
        assert not hasattr(widget, "via_debug_gradient_map_checkbox")
        assert not hasattr(widget, "noisy_traces_via_preset_button")
        assert not hasattr(widget, "blurred_via_preset_button")
        widget.resize(1280, 900)
        widget.show()
        widget.bright_via_basics_form.activate()
        basic_single_editors = (
            widget.via_search_mode_combo,
            widget.via_heuristic_polarity_combo,
            widget.via_diameter_size_mode_combo,
            widget.bright_via_diameter_fixed_spin,
            widget.bright_via_min_final_score_spin,
            widget.bright_via_nms_distance_spin,
        )
        assert {editor.width() for editor in basic_single_editors} == {
            widget.bright_via_basics_editor_width
        }
        basic_fields = basic_single_editors + (
            widget.bright_via_diameter_range_widget,
            widget.via_white_range_widget,
            widget.via_black_range_widget,
            widget.via_preset_widget,
            widget.reset_bright_via_button,
        )
        laid_out_x_positions = {
            field.geometry().x()
            for field in basic_fields
            if field.geometry().x() > 0
        }
        assert len(laid_out_x_positions) == 1
        assert all(
            widget.bright_via_basics_form.getWidgetPosition(field)[1]
            == QFormLayout.ItemRole.FieldRole
            for field in basic_fields
        )
        basic_labels = [
            widget.bright_via_basics_form.labelForField(field)
            for field in basic_fields
        ]
        assert {
            label.width()
            for label in basic_labels
            if label is not None
        } == {widget.bright_via_basics_label_width}

        widget.via_search_mode_combo.setCurrentIndex(widget.via_search_mode_combo.findData("template"))
        app.processEvents()
        assert widget.via_heuristic_polarity_combo.isHidden()

        widget.via_search_mode_combo.setCurrentIndex(widget.via_search_mode_combo.findData("hybrid"))
        app.processEvents()
        assert not widget.via_heuristic_polarity_combo.isHidden()
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


def test_template_table_reorders_all_template_metadata() -> None:
    app = QApplication.instance() or QApplication([])
    widget = PolygonExtractionWidget()
    try:
        assert widget.recognition_mode_label.text() == "Распознавание"
        assert [widget.recognition_mode_combo.itemText(i) for i in range(3)] == [
            "Отключено",
            "Проводники",
            "Контакты",
        ]
        widget._via_template_images = [np.full((5, 5), value, dtype=np.uint8) for value in range(1, 6)]
        widget._via_template_min_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        widget._via_template_diameters = [10, 20, 30, 40, 50]
        widget._refresh_via_template_list()

        widget.via_template_table.cellWidget(3, 0).setValue(2)
        app.processEvents()

        assert [int(image[0, 0]) for image in widget._via_template_images] == [1, 4, 2, 3, 5]
        assert widget._via_template_min_scores == [0.1, 0.4, 0.2, 0.3, 0.5]
        assert widget._via_template_diameters == [10, 40, 20, 30, 50]
        assert [widget.via_template_table.cellWidget(row, 0).value() for row in range(5)] == [1, 2, 3, 4, 5]
    finally:
        widget.close()
        widget.deleteLater()


def test_template_table_empty_hint_and_inherited_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    widget = PolygonExtractionWidget()
    try:
        empty_hint = widget.via_template_table.cellWidget(0, 0)
        assert empty_hint.text() == "Чтобы добавить шаблон удерживайте Ctrl и выделите область"
        assert widget.via_template_table.rowSpan(0, 0) == 1
        assert widget.via_template_table.columnSpan(0, 0) == 5
        assert not hasattr(widget, "add_via_template_button")

        source_image = np.arange(400, dtype=np.uint8).reshape(20, 20)
        monkeypatch.setattr(widget._workspace, "current_display_image", lambda: source_image)
        widget._on_editor_image_region_selected(1, 2, 5, 6)

        assert widget._via_template_min_scores == [0.6]
        assert widget._via_template_diameters == [8]

        widget._set_via_template_similarity(0, 0.73)
        widget._set_via_template_diameter(0, 12)
        widget._on_editor_image_region_selected(8, 9, 4, 4)

        assert widget._via_template_min_scores == [0.73, 0.73]
        assert widget._via_template_diameters == [12, 12]
        preview = widget.via_template_table.cellWidget(0, 1)
        assert (preview.width(), preview.height()) == (60, 60)
        remove_button = widget.via_template_table.cellWidget(0, 4)
        assert remove_button.sizePolicy().horizontalPolicy().name == "Expanding"
        assert remove_button.sizePolicy().verticalPolicy().name == "Expanding"
    finally:
        widget.close()
        widget.deleteLater()


def test_ctrl_template_selection_is_enabled_only_for_contact_template_modes() -> None:
    app = QApplication.instance() or QApplication([])
    widget = PolygonExtractionWidget()
    try:
        widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("via"))
        widget.via_search_mode_combo.setCurrentIndex(widget.via_search_mode_combo.findData("template"))
        app.processEvents()
        assert widget.polygon_editor._ctrl_image_region_selection_enabled

        widget.via_search_mode_combo.setCurrentIndex(widget.via_search_mode_combo.findData("hybrid"))
        app.processEvents()
        assert widget.polygon_editor._ctrl_image_region_selection_enabled

        widget.via_search_mode_combo.setCurrentIndex(widget.via_search_mode_combo.findData("heuristic"))
        app.processEvents()
        assert not widget.polygon_editor._ctrl_image_region_selection_enabled

        widget.via_search_mode_combo.setCurrentIndex(widget.via_search_mode_combo.findData("template"))
        widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("conductors"))
        app.processEvents()
        assert not widget.polygon_editor._ctrl_image_region_selection_enabled
    finally:
        widget.close()
        widget.deleteLater()


def test_contact_template_click_explains_template_number_and_similarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = QApplication.instance() or QApplication([])
    widget = PolygonExtractionWidget()
    try:
        widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("via"))
        app.processEvents()
        assert widget.debug_candidates_checkbox.isChecked()
        assert widget._via_debug_inspection_enabled()

        polygon = PolygonData(
            id=1,
            points=[(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)],
            area=100.0,
            perimeter=40.0,
            bbox=(10, 10, 10, 10),
            category="via",
            shape_hint="box",
        )
        widget._workspace._current_state = ImageProcessingState(
            image_path="sample.png",
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[polygon],
            debug_candidates=[
                ContourDebugCandidate(
                    contour_index=0,
                    bbox=(10, 10, 10, 10),
                    accepted=True,
                    reason="accepted:template",
                    source="template",
                    score=87.3,
                    template_index=1,
                )
            ],
        )
        shown: list[str] = []
        monkeypatch.setattr(
            widget,
            "_show_nonblocking_via_debug_message",
            lambda _title, message: shown.append(str(message)),
        )

        widget._on_via_debug_requested(polygon)

        assert len(shown) == 1
        assert "Метод: По шаблону" in shown[0]
        assert "Номер шаблона: 2" in shown[0]
        assert "Похожесть: 0,873" in shown[0]
        assert "legacy" not in shown[0]
        assert "Оценка:" not in shown[0]
        assert "Округлость:" not in shown[0]
        assert "Причина:" not in shown[0]
    finally:
        widget.close()
        widget.deleteLater()


def test_heuristic_contact_click_shows_measured_features_and_expert_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = QApplication.instance() or QApplication([])
    widget = PolygonExtractionWidget()
    try:
        polygon = PolygonData(
            id=1,
            points=[(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)],
            area=100.0,
            perimeter=40.0,
            bbox=(10, 10, 10, 10),
            category="via",
            shape_hint="box",
        )
        widget._workspace._current_state = ImageProcessingState(
            image_path="sample.png",
            source_image=np.zeros((32, 32), dtype=np.uint8),
            polygons=[polygon],
            debug_candidates=[
                ContourDebugCandidate(
                    contour_index=0,
                    bbox=(10, 10, 10, 10),
                    accepted=True,
                    reason="accepted:heuristic",
                    source="heuristic",
                    score=81.25,
                    metrics={
                        "center_brightness": 214.0,
                        "binarization_threshold": 176.0,
                        "contrast": 42.5,
                        "prominence": 51.0,
                        "diameter": 10.0,
                        "equivalent_diameter": 9.7,
                        "center_drift": 0.4,
                        "compactness": 0.83,
                        "circularity": 0.91,
                        "aspect": 1.08,
                        "line_coherence": 0.17,
                        "edge_snr": 3.2,
                        "edge_sharpness": 0.88,
                        "border_imbalance": 0.06,
                        "line_likeness": 0.12,
                        "contribution_contrast": 22.0,
                        "contribution_prominence": 18.0,
                        "contribution_size": 19.0,
                        "contribution_compactness": 12.0,
                        "contribution_roundness": 9.0,
                        "contribution_balance": 9.5,
                        "penalty_line": 2.4,
                        "penalty_border": 1.2,
                        "final_score": 81.25,
                    },
                )
            ],
        )
        shown: list[str] = []
        monkeypatch.setattr(
            widget,
            "_show_nonblocking_via_debug_message",
            lambda _title, message: shown.append(str(message)),
        )

        widget._on_via_debug_requested(polygon)

        assert len(shown) == 1
        assert "Измеренные параметры контакта:" in shown[0]
        assert "Яркость центра: 214,0 из 255" in shown[0]
        assert "Компактность: 0,830" in shown[0]
        assert "Округлость формы: 0,910 (минимум" in shown[0]
        assert "Направленность границ: 0,170" in shown[0]
        assert "Вклад в итоговую оценку:" in shown[0]
        assert "Контраст: +22,00 (вес" in shown[0]
        assert "Применённые настройки генерации кандидатов:" in shown[0]
        assert "Процентили пиков" in shown[0]
    finally:
        widget.close()
        widget.deleteLater()


def test_via_debug_windows_are_nonmodal_and_can_stay_open_together() -> None:
    app = QApplication.instance() or QApplication([])
    widget = PolygonExtractionWidget()
    try:
        widget.show()
        first = widget._show_nonblocking_via_debug_message("Via 1", "First")
        second = widget._show_nonblocking_via_debug_message("Via 2", "Second")
        app.processEvents()

        assert first.windowModality() == Qt.WindowModality.NonModal
        assert second.windowModality() == Qt.WindowModality.NonModal
        assert not first.isModal()
        assert not second.isModal()
        assert first.isVisible()
        assert second.isVisible()
        assert first.pos() != second.pos()
        assert widget._open_via_debug_dialogs == [first, second]

        first.close()
        app.processEvents()
        assert widget._open_via_debug_dialogs == [second]
        second.close()
        app.processEvents()
        assert widget._open_via_debug_dialogs == []
    finally:
        for dialog in list(getattr(widget, "_open_via_debug_dialogs", [])):
            dialog.close()
        widget.close()
        widget.deleteLater()


def test_via_profiles_set_expected_parameters() -> None:
    presets = built_in_via_presets("ru")
    assert set(presets) == {
        "Стандартный",
        "Строгий",
        "Чувствительный",
        "Шумное изображение",
        "Размытые контакты",
    }
    assert all(
        payload and all(key.startswith("heuristic_") for key in payload)
        for payload in presets.values()
    )
    assert (
        presets["Строгий"]["heuristic_min_center_contrast"]
        > presets["Чувствительный"]["heuristic_min_center_contrast"]
    )


def test_saved_json_contains_pipeline_all_recognition_settings_and_via_presets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    app = QApplication.instance() or QApplication([])
    widget = PolygonExtractionWidget()
    path = tmp_path / "recognition.json"
    try:
        widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("via"))
        widget._via_template_images = [np.full((5, 7), 180, dtype=np.uint8)]
        widget._via_template_min_scores = [0.71]
        widget._via_template_diameters = [9]
        widget.heuristic_w_contrast_spin.setValue(31.0)
        widget._user_via_presets = {"My expert": widget._current_via_preset_payload()}
        monkeypatch.setattr(
            "contour.widget_parts.pipeline_actions_mixin.QFileDialog.getSaveFileName",
            lambda *_args, **_kwargs: (str(path), "JSON"),
        )

        widget._save_pipeline_json()

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["format"] == "contour-recognition-settings"
        assert payload["version"] == 2
        assert isinstance(payload["pipeline"]["steps"], list)
        recognition = payload["recognition_settings"]
        assert recognition == widget._current_contour_settings().to_dict()
        assert recognition["via_template_images"]
        assert recognition["via_template_min_scores"] == [0.71]
        assert recognition["via_template_diameters"] == [9]
        expert = payload["via_expert_presets"]["user"]["My expert"]
        assert expert["heuristic_w_contrast"] == 31.0
        assert all(key.startswith("heuristic_") for key in expert)

        widget.heuristic_w_contrast_spin.setValue(7.0)
        widget._via_template_images = []
        widget._via_template_min_scores = []
        widget._via_template_diameters = []
        widget._user_via_presets = {}
        monkeypatch.setattr(
            "contour.widget_parts.pipeline_actions_mixin.QFileDialog.getOpenFileName",
            lambda *_args, **_kwargs: (str(path), "JSON"),
        )
        widget._load_pipeline_json()
        assert widget.heuristic_w_contrast_spin.value() == 31.0
        assert len(widget._via_template_images) == 1
        assert widget._via_template_min_scores == [0.71]
        assert widget._via_template_diameters == [9]
        assert "My expert" in widget._user_via_presets
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_via_preset_json_contains_full_snapshot_but_applies_only_expert_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    app = QApplication.instance() or QApplication([])
    widget = PolygonExtractionWidget()
    path = tmp_path / "expert.json"
    try:
        widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("via"))
        widget.heuristic_w_contrast_spin.setValue(34.0)
        monkeypatch.setattr(
            "contour.widget_parts.pipeline_mixin.QInputDialog.getText",
            lambda *_args, **_kwargs: ("My expert", True),
        )
        monkeypatch.setattr(
            "contour.widget_parts.pipeline_mixin.QFileDialog.getSaveFileName",
            lambda *_args, **_kwargs: (str(path), "JSON"),
        )

        widget._save_current_via_preset()

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["format"] == "contour-via-expert-preset"
        assert payload["name"] == "My expert"
        assert payload["recognition_settings"] == widget._current_contour_settings().to_dict()
        assert payload["expert_settings"]["heuristic_w_contrast"] == 34.0
        assert all(key.startswith("heuristic_") for key in payload["expert_settings"])

        widget.heuristic_w_contrast_spin.setValue(7.0)
        widget.via_search_mode_combo.setCurrentIndex(widget.via_search_mode_combo.findData("template"))
        widget.via_heuristic_polarity_combo.setCurrentIndex(
            widget.via_heuristic_polarity_combo.findData("dark")
        )
        widget.bright_via_diameter_fixed_spin.setValue(13)
        monkeypatch.setattr(
            "contour.widget_parts.pipeline_actions_mixin.QFileDialog.getOpenFileName",
            lambda *_args, **_kwargs: (str(path), "JSON"),
        )
        widget._load_pipeline_json()

        assert widget.heuristic_w_contrast_spin.value() == 34.0
        assert widget.via_search_mode_combo.currentData() == "template"
        assert widget.via_heuristic_polarity_combo.currentData() == "dark"
        assert widget.bright_via_diameter_fixed_spin.value() == 13
        assert widget.bright_via_advanced_outer.isChecked()
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
