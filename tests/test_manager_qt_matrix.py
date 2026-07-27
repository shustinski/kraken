from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QContextMenuEvent
from PyQt6.QtWidgets import QApplication

from kraken_manager.presentation.qt import (
    FrameCellData,
    FrameMatrixView,
    FrameRect,
    FrameSelection,
    GridOrientation,
    MatrixLod,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_large_sparse_matrix_virtualizes_visible_tiles(qapp):
    view = FrameMatrixView(10_000_000, 1)
    view.resize(900, 480)
    view.show()
    qapp.processEvents()
    view.zoom_to_fit()
    qapp.processEvents()

    assert view.materialized_cell_count() == 0
    assert view.visible_item_count() < 20
    assert view.lod_level() is MatrixLod.OVERVIEW


def test_matrix_coordinates_follow_project_y_orientation(qapp):
    view = FrameMatrixView(4, 3, GridOrientation.Y_DOWN)
    top_left = view.scene_rect_for_frame(1, 1).center()
    assert view.frame_at_scene_pos(top_left) == (1, 1)

    view.set_orientation(GridOrientation.Y_UP)

    top_left = view.scene_rect_for_frame(1, 3).center()
    assert view.frame_at_scene_pos(top_left) == (1, 3)
    assert view.scene_rect_for_frame(1, 1).top() > top_left.y()


def test_matrix_uses_sparse_presentation_data_and_compact_selection(qapp):
    view = FrameMatrixView(1_000, 1_000)
    cell = FrameCellData(
        20,
        30,
        status="in_review",
        performer_color="#ff00ff",
        performer_initials="AK",
        payload={"opaque": True},
    )
    view.set_cells([cell])
    selection = FrameSelection((FrameRect(10, 20, 30, 40),))
    view.set_selection(selection)

    assert view.materialized_cell_count() == 1
    assert view.cell_data(20, 30) is cell
    assert view.cell_data(21, 30).status == "empty"
    assert view.selection() == selection
    assert selection.contains(20, 30)


def test_context_menu_emits_frame_and_current_selection(qapp):
    view = FrameMatrixView(3, 3)
    view.resize(360, 280)
    view.show()
    qapp.processEvents()
    frame_center = view.mapFromScene(view.scene_rect_for_frame(2, 2).center())
    emitted = []
    view.contextMenuRequested.connect(lambda context, position: emitted.append((context, position)))
    event = QContextMenuEvent(
        QContextMenuEvent.Reason.Mouse,
        frame_center,
        view.viewport().mapToGlobal(frame_center),
    )

    QApplication.sendEvent(view.viewport(), event)

    assert emitted
    context, _position = emitted[-1]
    assert (context.x, context.y) == (2, 2)
    assert context.selection == FrameSelection.single(2, 2)


def test_frame_selection_expansion_can_be_bounded():
    selection = FrameSelection((FrameRect(1, 1, 100, 100),))

    with pytest.raises(ValueError, match="more than 10"):
        tuple(selection.coordinates(maximum=10))
