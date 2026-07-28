from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from kraken_manager.workspace import scan_layer_source
from kraken_manager.presentation.qt import LayerCreationDialog


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _dialog(maximum_frames: int = 4) -> LayerCreationDialog:
    return LayerCreationDialog(
        maximum_frames=maximum_frames,
        scanner=scan_layer_source,
    )


def test_name_and_source_are_mandatory_in_both_modes(qapp, tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (2, 2), "white").save(images / "frame_0.jpg")
    dialog = _dialog()

    assert not dialog.ok_button.isEnabled()
    dialog.tabs.setCurrentWidget(dialog.manual_tab)
    dialog.manual_images.edit.setText(str(images))
    assert not dialog.ok_button.isEnabled()
    dialog.name_edit.setText("Metal")
    assert dialog.ok_button.isEnabled()

    dialog.tabs.setCurrentWidget(dialog.disk_tab)
    assert not dialog.ok_button.isEnabled()
    dialog._scan_succeeded(scan_layer_source(images, maximum_frames=4))
    assert dialog.ok_button.isEnabled()


def test_jpg_scan_hides_conversion_but_keeps_reflections(qapp, tmp_path: Path) -> None:
    images = tmp_path / "jpg"
    images.mkdir()
    Image.new("RGB", (2, 2), "white").save(images / "frame_0.jpg")
    dialog = _dialog()
    dialog.show()
    initial_tab_height = dialog.tabs.height()
    dialog._scan_succeeded(scan_layer_source(images, maximum_frames=4))

    assert dialog.jpg_count_label.text() == "1"
    assert dialog.bmp_count_label.text() == "0"
    assert dialog.tabs.height() > initial_tab_height
    assert not dialog.jpg_radio.isVisible()
    assert not dialog.png_radio.isVisible()
    assert dialog.flip_horizontal.isVisible()
    assert dialog.flip_vertical.isVisible()


def test_bmp_scan_exposes_exactly_one_conversion_format_and_advanced_settings(
    qapp, tmp_path: Path
) -> None:
    images = tmp_path / "bmp"
    images.mkdir()
    Image.new("RGB", (2, 2), "white").save(images / "frame_0.bmp")
    dialog = _dialog()
    dialog.show()
    dialog._scan_succeeded(scan_layer_source(images, maximum_frames=4))

    assert dialog.bmp_count_label.text() == "1"
    assert dialog.jpg_radio.isVisible()
    assert dialog.png_radio.isVisible()
    assert dialog.jpg_radio.isChecked()
    dialog.png_radio.setChecked(True)
    assert dialog.format_settings.currentIndex() == 1
    dialog.png_optimize.setChecked(True)
    assert not dialog.png_compression.isEnabled()


def test_mixed_images_show_blocking_scan_error(qapp, tmp_path: Path) -> None:
    images = tmp_path / "mixed"
    images.mkdir()
    Image.new("RGB", (2, 2), "white").save(images / "frame_0.jpg")
    Image.new("RGB", (2, 2), "white").save(images / "frame_1.bmp")
    dialog = _dialog()
    dialog.name_edit.setText("Metal")
    dialog._scan_succeeded(scan_layer_source(images, maximum_frames=4))

    assert not dialog.ok_button.isEnabled()
    assert "JPG" in dialog.validation_label.text()
    assert "BMP" in dialog.validation_label.text()
