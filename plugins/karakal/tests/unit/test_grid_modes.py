"""Acceptance coverage for the four grid-inspection data modes."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QSettings

from karakal.app.main_window import KarakalMainWindow, QtUpdateController
from karakal.core.domain import BuildOptions, BuildResult, FrameRecord
from karakal.core.grid_anomaly import GridDamageAnalysisConfig, detect_grid_cell_anomalies
from karakal.core.grid_hints import cell_low_confidence, possible_missed_regions, source_cell_rule, source_mask_mismatches
from karakal.core.image_io import _grayscale_array_to_qimage
from karakal.ui.details_dialog import ExtendFrameDetailsDialog
from karakal.ui.grid_tuning_dialog import GridTuningDialog
from karakal.ui.i18n import Translator
from karakal.ui.ui_constants import GRID_INSPECTION_PRESET_VALUES


def _outline() -> np.ndarray:
    image = np.zeros((360, 480), dtype=np.uint8)
    for row in range(4):
        for col in range(5):
            x, y = 24 + col * 90, 20 + row * 84
            cv2.rectangle(image, (x, y), (x + 48, y + 44), 255, 2)
    return image


def _growing() -> np.ndarray:
    image = np.zeros((420, 980), dtype=np.uint8)
    base = 28
    gap = 6
    widths = [int(round(base * (1.0 + 0.6 * col / 7))) for col in range(8)]
    for row in range(6):
        x = 10
        y = 10 + row * (base + gap)
        for width in widths:
            cv2.rectangle(image, (x, y), (x + width, y + base), 255, 2)
            x += width + gap
    return image


def _calibrated(**sliders: int) -> GridDamageAnalysisConfig:
    values = dict(GRID_INSPECTION_PRESET_VALUES["balanced"])
    values.update(sliders)
    return GridDamageAnalysisConfig(
        cell_representation="binary",
        scoring_mode="calibrated",
        blur_radius=1,
        min_contour_area=80.0,
        min_cell_size=8,
        fill_sensitivity=int(values["fill_sensitivity"]),
        debris_sensitivity=int(values["debris_sensitivity"]),
        geometry_sensitivity=int(values["geometry_sensitivity"]),
        merge_sensitivity=int(values["merge_sensitivity"]),
        min_grid_candidate_cells=4,
    )


@pytest.mark.parametrize("mode", ("A", "B", "C", "D"))
def test_mask_defects_are_the_same_in_every_data_mode(mode: str) -> None:
    image = _outline()
    cv2.rectangle(image, (26, 22), (70, 62), 255, -1)
    config = _calibrated()
    if mode in {"B", "D"}:
        confidence = np.full(image.shape, 0.9, dtype=np.float32)
        confidence[30:50, 30:50] = 0.4
    else:
        confidence = None
    result = detect_grid_cell_anomalies(image, config=config, confidence_map=confidence)
    reasons = {reason for cell in result.cells for reason in cell.reasons}
    assert "filled_cell" in reasons or "partial_filled_cell" in reasons
    if mode in {"C", "D"}:
        from karakal.core.grid_ground_truth import match_masks

        matched = match_masks(image, _outline())
        assert matched["components"]


@pytest.mark.parametrize("preset", ("soft", "balanced", "strict"))
def test_mode_a_presets_find_defects_and_keep_clean_grids(preset: str) -> None:
    values = GRID_INSPECTION_PRESET_VALUES[preset]
    config = _calibrated(**values)
    clean = detect_grid_cell_anomalies(_outline(), config=config)
    assert {reason for cell in clean.cells for reason in cell.reasons} == set()
    growing = detect_grid_cell_anomalies(_growing(), config=config)
    assert {reason for cell in growing.cells for reason in cell.reasons} == set()
    filled = _outline()
    cv2.rectangle(filled, (26, 22), (70, 62), 255, -1)
    assert any(
        reason in {"filled_cell", "partial_filled_cell"}
        for cell in detect_grid_cell_anomalies(filled, config=config).cells
        for reason in cell.reasons
    )


def test_mode_a_windows_hide_markup_controls(tmp_path, qtbot, monkeypatch) -> None:
    monkeypatch.setattr(QtUpdateController, "check_for_updates", lambda self, manual=False: None)
    mask_path = tmp_path / "mask.png"
    assert _grayscale_array_to_qimage(np.full((48, 48), 255, dtype=np.uint8)).save(str(mask_path))
    settings = QSettings(str(tmp_path / "k.ini"), QSettings.Format.IniFormat)
    window = KarakalMainWindow(settings=settings)
    qtbot.addWidget(window)
    record = FrameRecord("frame-1", "Frame 1", model_mask_paths={"model": str(mask_path)})
    dialog = ExtendFrameDetailsDialog(
        record,
        BuildResult(records=(record,), options=BuildOptions()),
        session_view_state={"preferred_model_id": "model", "result_kind": "grid_cell_defects"},
        allowed_result_kinds=("grid_cell_defects",),
    )
    dialog.show()
    translator = Translator("ru")
    tuning = GridTuningDialog(translator.tr, dict(GRID_INSPECTION_PRESET_VALUES["balanced"]), parent=dialog)
    tuning.show()
    assert hasattr(tuning, "_markup_button")
    assert tuning._markup_button.isEnabled() is False
    assert tuning._markup_note.isVisible()
    tuning.close()
    dialog.close()
    window.close()


def test_mixed_confidence_matrix_payload_is_safe() -> None:
    image = _outline()
    with_conf = detect_grid_cell_anomalies(
        image,
        config=_calibrated(),
        confidence_map=np.full(image.shape, 0.8, dtype=np.float32),
    )
    without = detect_grid_cell_anomalies(image, config=_calibrated())
    assert {reason for cell in with_conf.cells for reason in cell.reasons} == {
        reason for cell in without.cells for reason in cell.reasons
    }


def test_optional_hints_stay_off_the_mask_layer() -> None:
    assert cell_low_confidence(0.2, 0.5) is True
    assert cell_low_confidence(0.9, 0.0) is False
    probability = np.zeros((40, 40), dtype=np.float32)
    probability[10:20, 10:20] = 0.8
    mask = np.zeros((40, 40), dtype=np.uint8)
    assert possible_missed_regions(probability, mask)
    source = np.full((40, 40), 40, dtype=np.uint8)
    source[5:15, 5:15] = 200
    truth = np.zeros((40, 40), dtype=np.uint8)
    truth[5:15, 5:15] = 255
    rule = source_cell_rule(source, truth)
    assert rule["usable"] == 1.0
    network = np.zeros((40, 40), dtype=np.uint8)
    assert source_mask_mismatches(source, network, rule)
