from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtWidgets import QApplication

from kraken_core.frame_matrix.qt import FrameMatrixView


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_default_geometry_is_square_kraken_cell(qapp):
    view = FrameMatrixView(2, 2)

    assert view.cell_size() == (view.CELL_SIZE, view.CELL_SIZE)
    assert view.cell_pitch() == (view.CELL_PITCH, view.CELL_PITCH)


def test_rectangular_cells_with_overlap(qapp):
    view = FrameMatrixView(3, 2)
    view.set_cell_geometry(64, 48, 8, 6)

    assert view.cell_size() == (64.0, 48.0)
    assert view.cell_pitch() == (56.0, 42.0)
    assert view.scene_rect_for_frame(2, 1) == QRectF(56.0, 0.0, 64.0, 48.0)
    assert view.scene_rect_for_frame(1, 2) == QRectF(0.0, 42.0, 64.0, 48.0)
    scene = view.sceneRect()
    assert scene.width() == 3 * 56.0 + 8.0
    assert scene.height() == 2 * 42.0 + 6.0
    # The overlapped strip belongs to the later cell, which is painted on top.
    assert view.frame_at_scene_pos(QPointF(60.0, 10.0)) == (2, 1)
    assert view.frame_at_scene_pos(QPointF(10.0, 45.0)) == (1, 2)


def test_overlap_is_clamped_below_cell_size(qapp):
    view = FrameMatrixView(2, 2)
    view.set_cell_geometry(64, 48, 100, 100)

    assert view.cell_pitch() == (1.0, 1.0)
