"""Grid details image stack: source under confidence under the network mask."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QGraphicsPixmapItem

from karakal.core.domain import BuildOptions, BuildResult, FrameRecord
from karakal.core.grid_anomaly import GridCellAnalysisResult, GridFrameAnalysisResult
from karakal.core.image_io import _grayscale_array_to_qimage, load_grayscale_image
from karakal.ui.details_dialog import ExtendFrameDetailsDialog


def _dialog(tmp_path, qtbot, *, source: bool, confidence: bool, mask: np.ndarray | None = None):
    original = tmp_path / "original.png"
    mask_path = tmp_path / "mask.png"
    confidence_path = tmp_path / "confidence.png"
    source_gray = np.full((32, 32), 40, dtype=np.uint8)
    if source:
        assert _grayscale_array_to_qimage(source_gray).save(str(original))
    mask_values = np.zeros((32, 32), dtype=np.uint8) if mask is None else mask
    mask_values[4:12, 4:12] = 255
    assert _grayscale_array_to_qimage(mask_values).save(str(mask_path))
    if confidence:
        assert _grayscale_array_to_qimage(np.full((32, 32), 250, dtype=np.uint8)).save(str(confidence_path))
    record = FrameRecord(
        "frame-1",
        "Frame 1",
        original_path=str(original) if source else "",
        model_mask_paths={"model": str(mask_path)},
        model_prob_paths={"model": str(confidence_path)} if confidence else {},
    )
    dialog = ExtendFrameDetailsDialog(
        record,
        BuildResult(records=(record,), options=BuildOptions()),
        session_view_state={"preferred_model_id": "model", "result_kind": "grid_cell_defects"},
        allowed_result_kinds=("grid_cell_defects",),
        grid_inspection_source_path=str(original) if source else str(mask_path),
    )
    qtbot.addWidget(dialog)
    dialog._payload["model_source_grays"] = {"model": mask_values.astype(np.float32) / 255.0}
    if confidence:
        dialog._payload["model_confidence_output_available"] = {"model": True}
        dialog._payload["model_output_probabilities"] = {"model": np.full((32, 32), 0.99, dtype=np.float32)}
    cell = GridCellAnalysisResult(
        row=0,
        col=0,
        bbox=(8, 8, 6, 6),
        centroid=(11.0, 11.0),
        contour_id=1,
        status="broken",
        score=0.8,
        reasons=("broken_geometry",),
    )
    dialog._grid_inspection_result = GridFrameAnalysisResult(
        frame_id="frame-1",
        frame_path="",
        image_width=32,
        image_height=32,
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=1,
        detected_cells=1,
        normal_cells=0,
        suspicious_cells=0,
        broken_cells=1,
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.4,
        severity_level="WARN",
        grid_detected=True,
        per_cell_results=(cell,),
    )
    dialog._refresh_result_kind_options("grid_cell_defects")
    dialog._refresh_scene(reset_view=False)
    return dialog


def _uncheck_all(dialog) -> None:
    for checkbox in dialog._grid_hotkey_layers():
        checkbox.setChecked(False)


def test_all_layers_off_hides_pixmaps_and_boxes(tmp_path, qtbot) -> None:
    dialog = _dialog(tmp_path, qtbot, source=True, confidence=True)
    _uncheck_all(dialog)
    dialog._update_layer_states()
    dialog._rebuild_grid_mark_items()

    for item in (
        dialog.original_item,
        dialog.second_source_item,
        dialog.first_source_item,
        dialog.result_item,
    ):
        assert isinstance(item, QGraphicsPixmapItem)
        assert not item.isVisible()
    assert dialog._grid_mark_items == []


def test_only_network_result_shows_the_mask(tmp_path, qtbot) -> None:
    dialog = _dialog(tmp_path, qtbot, source=True, confidence=True)
    dialog.grid_confidence_visible.setChecked(False)
    dialog.grid_source_visible.setChecked(False)
    dialog.grid_network_visible.setChecked(True)
    _uncheck_all(dialog)
    dialog.grid_network_visible.setChecked(True)
    dialog._update_layer_states()

    assert dialog.first_source_item.isVisible()
    assert not dialog.original_item.isVisible()
    assert not dialog.second_source_item.isVisible()
    assert not dialog.result_item.isVisible()


def test_grid_image_z_order_is_source_confidence_mask_then_marks(tmp_path, qtbot) -> None:
    dialog = _dialog(tmp_path, qtbot, source=True, confidence=True)
    assert dialog.original_item.zValue() < dialog.second_source_item.zValue()
    assert dialog.second_source_item.zValue() < dialog.first_source_item.zValue()
    assert dialog.first_source_item.zValue() < dialog._grid_mark_items[0].zValue()


def test_zero_mask_pixel_leaves_the_lower_layer_color(tmp_path, qtbot) -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[0:8, 0:8] = 255
    dialog = _dialog(tmp_path, qtbot, source=True, confidence=False, mask=mask)
    dialog.grid_network_mode.setCurrentIndex(dialog.grid_network_mode.findData("binary"))
    source = dialog.original_item.pixmap().toImage().convertToFormat(QImage.Format.Format_ARGB32)
    network = dialog.first_source_item.pixmap().toImage().convertToFormat(QImage.Format.Format_ARGB32)
    assert network.pixelColor(20, 20).alpha() == 0
    assert network.pixelColor(2, 2).alpha() == 255
    composed = QImage(source)
    painter = QPainter(composed)
    painter.setOpacity(dialog.grid_network_opacity.value() / 100.0)
    painter.drawImage(0, 0, network)
    painter.end()
    assert composed.pixelColor(20, 20).red() == source.pixelColor(20, 20).red()


def test_missing_confidence_disables_without_forgetting_the_choice(tmp_path, qtbot) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    without = _dialog(tmp_path / "a", qtbot, source=True, confidence=False)
    assert without.grid_confidence_visible.isChecked()
    assert not without.grid_confidence_visible.isEnabled()
    assert without.grid_confidence_visible.toolTip() == without._t("details.layer_missing")
    stored = without._build_view_settings_payload()
    with_file = _dialog(tmp_path / "b", qtbot, source=True, confidence=True)
    with_file._session_view_state.update(stored)
    with_file._restore_view_settings()
    with_file._apply_grid_layer_availability()
    assert with_file.grid_confidence_visible.isEnabled()
    assert with_file.grid_confidence_visible.isChecked()


def test_layer_choices_survive_a_scene_refresh(tmp_path, qtbot) -> None:
    dialog = _dialog(tmp_path, qtbot, source=True, confidence=True)
    dialog.grid_source_visible.setChecked(False)
    dialog.grid_network_opacity.setValue(25)
    dialog._store_view_settings()
    dialog._refresh_scene(reset_view=False)
    assert not dialog.grid_source_visible.isChecked()
    assert dialog.grid_network_opacity.value() == 25
    assert not dialog.original_item.isVisible()
    assert dialog.first_source_item.opacity() == 0.25


def _drop_layer_caches(dialog) -> None:
    dialog._derived_cache.clear()
    dialog._overlay_cache.clear()


def test_original_pixmap_matches_the_gray_file(tmp_path, qtbot) -> None:
    dialog = _dialog(tmp_path, qtbot, source=True, confidence=True)
    assert dialog.grid_source_mode.currentData() == "original"
    assert dialog.grid_network_mode.currentData() == "original"
    loaded = load_grayscale_image(Path(dialog._record.original_path))
    image = dialog.original_item.pixmap().toImage()
    for y, x in ((0, 0), (4, 9), (31, 31)):
        color = image.pixelColor(x, y)
        value = int(loaded[y, x])
        assert (color.red(), color.green(), color.blue(), color.alpha()) == (value, value, value, 255)
    network = dialog.first_source_item.pixmap().toImage()
    assert network.pixelColor(20, 20).alpha() == 255
    assert network.pixelColor(20, 20).red() == network.pixelColor(20, 20).green() == network.pixelColor(20, 20).blue()


def test_layer_modes_are_independent_and_survive_a_frame_change(tmp_path, qtbot) -> None:
    dialog = _dialog(tmp_path, qtbot, source=True, confidence=True)
    dialog.grid_network_threshold.setValue(70)
    dialog.grid_confidence_threshold.setValue(40)
    dialog.grid_confidence_invert.setChecked(True)
    dialog.grid_network_mode.setCurrentIndex(dialog.grid_network_mode.findData("original"))
    dialog.grid_confidence_mode.setCurrentIndex(dialog.grid_confidence_mode.findData("heatmap"))
    network = dialog.first_source_item.pixmap().toImage().pixelColor(2, 2)
    confidence = dialog.second_source_item.pixmap().toImage().pixelColor(0, 0)
    assert network.red() == network.green() == network.blue()
    assert confidence.alpha() == 0 or confidence.red() != confidence.green()
    stored = dialog._build_view_settings_payload()
    (tmp_path / "next").mkdir()
    other = _dialog(tmp_path / "next", qtbot, source=True, confidence=True)
    other._session_view_state.clear()
    other._session_view_state.update(stored)
    other._restore_view_settings()
    assert other.grid_network_mode.currentData() == "original"
    assert other.grid_confidence_mode.currentData() == "heatmap"
    assert other.grid_network_threshold.value() == 70
    assert other.grid_confidence_threshold.value() == 40
    assert other.grid_confidence_invert.isChecked()
    assert other.grid_source_mode.currentData() == "original"


def test_binary_threshold_and_invert(tmp_path, qtbot) -> None:
    dialog = _dialog(tmp_path, qtbot, source=True, confidence=True)
    confidence = np.full((32, 32), 20, dtype=np.uint8)
    confidence[0, 0] = 255
    assert _grayscale_array_to_qimage(confidence).save(str(tmp_path / "confidence.png"))
    _drop_layer_caches(dialog)
    dialog.grid_confidence_mode.setCurrentIndex(dialog.grid_confidence_mode.findData("binary"))
    dialog.grid_confidence_threshold.setValue(50)
    dialog.grid_confidence_invert.setChecked(False)
    image = dialog.second_source_item.pixmap().toImage().convertToFormat(QImage.Format.Format_ARGB32)
    assert image.pixelColor(4, 4).alpha() == 0
    assert image.pixelColor(0, 0).alpha() == 255
    assert image.pixelColor(0, 0).red() == 239
    dialog.grid_confidence_invert.setChecked(True)
    inverted = dialog.second_source_item.pixmap().toImage().convertToFormat(QImage.Format.Format_ARGB32)
    assert inverted.pixelColor(4, 4).alpha() == 255
    assert inverted.pixelColor(0, 0).alpha() == 0
    network = dialog.first_source_item.pixmap().toImage().pixelColor(6, 6)
    assert network.red() == network.green() == network.blue()


def test_heatmap_hides_confident_pixels_and_shows_the_frame_minimum(tmp_path, qtbot) -> None:
    dialog = _dialog(tmp_path, qtbot, source=True, confidence=True)
    confidence = np.full((32, 32), 247, dtype=np.uint8)
    confidence[0, 0] = 255
    confidence[31, 31] = 51
    assert _grayscale_array_to_qimage(confidence).save(str(tmp_path / "confidence.png"))
    _drop_layer_caches(dialog)
    dialog.grid_confidence_mode.setCurrentIndex(dialog.grid_confidence_mode.findData("heatmap"))
    dialog._refresh_scene(reset_view=False)
    image = dialog.second_source_item.pixmap().toImage().convertToFormat(QImage.Format.Format_ARGB32)
    assert image.pixelColor(0, 0).alpha() == 0
    assert image.pixelColor(31, 31).alpha() == 255
    assert "0.200" in dialog.grid_confidence_legend.text()
    assert "1.000" in dialog.grid_confidence_legend.text()
    assert dialog.original_item.zValue() < dialog.second_source_item.zValue() < dialog.first_source_item.zValue()
    assert dialog.first_source_item.zValue() < dialog._grid_mark_items[0].zValue()


def test_opacity_does_not_rebuild_pixmap_and_mode_does_not_reread(tmp_path, qtbot, monkeypatch) -> None:
    dialog = _dialog(tmp_path, qtbot, source=True, confidence=True)
    gray_key = next(key for key in dialog._overlay_cache if key[0] == "grid_layer_gray" and key[1] == "network")
    pixmap = dialog._overlay_cache[gray_key]
    dialog.grid_network_opacity.setValue(10)
    assert dialog._overlay_cache[gray_key] is pixmap
    assert dialog.first_source_item.opacity() == 0.1

    def _refuse_read(*_args, **_kwargs):
        raise AssertionError("mode change re-read the file")

    monkeypatch.setattr("karakal.ui.details_dialog.load_grayscale_image", _refuse_read)
    dialog.grid_network_mode.setCurrentIndex(dialog.grid_network_mode.findData("binary"))
    pixel = dialog.first_source_item.pixmap().toImage().convertToFormat(QImage.Format.Format_ARGB32).pixelColor(6, 6)
    assert pixel.alpha() == 255
    assert (pixel.red(), pixel.green(), pixel.blue()) == (80, 210, 255)
