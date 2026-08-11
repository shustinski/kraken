from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication

from kraken_manager.presentation.qt import GridDimensionsWidget, GridOrientation


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_grid_dimensions_are_in_frames_and_include_orientation(qapp):
    widget = GridDimensionsWidget(40, 25, GridOrientation.Y_UP)

    assert widget.target_size() == QSize(40, 25)
    assert widget.dimensions().orientation is GridOrientation.Y_UP
    assert widget.dimensions().frame_count == 1_000
    assert widget.orientation_combo.currentData() == "y_up"
    assert widget.width_unit_badge.text() == "кадр."
    assert "пиксел" not in widget.toolTip().lower()


def test_grid_dimensions_reports_cap_without_silently_clamping(qapp):
    widget = GridDimensionsWidget(10, 10, maximum_frames=120)
    widget.size_lock_button.setChecked(False)
    emissions: list[tuple[bool, str]] = []
    widget.validityChanged.connect(lambda valid, message: emissions.append((valid, message)))

    widget.width_spinbox.setValue(20)

    assert widget.frame_count() == 200
    assert not widget.is_valid()
    assert widget.validation_label.property("valid") is False
    assert "120" in widget.validation_message()
    assert emissions[-1][0] is False
    with pytest.raises(ValueError, match="120"):
        widget.validated_dimensions()


def test_grid_dimensions_are_unlimited_by_default(qapp):
    widget = GridDimensionsWidget(2_000_000, 2_000_000)

    assert widget.maximum_frames() is None
    assert widget.is_valid()
    assert widget.frame_count() == 4_000_000_000_000
    assert "без ограничения" in widget.validation_message()


def test_set_dimensions_publishes_one_complete_snapshot(qapp):
    widget = GridDimensionsWidget()
    emissions: list[tuple[int, int, str]] = []
    widget.dimensionsChanged.connect(lambda width, height, orientation: emissions.append((width, height, orientation)))

    widget.set_dimensions(32, 18, GridOrientation.Y_UP)

    assert widget.dimensions().width == 32
    assert widget.dimensions().height == 18
    assert emissions == [(32, 18, "y_up")]


@pytest.mark.parametrize("width,height", [(0, 1), (1, 0), (-1, 10)])
def test_grid_dimensions_reject_non_positive_axes(qapp, width, height):
    with pytest.raises(ValueError, match="positive"):
        GridDimensionsWidget(width, height)
