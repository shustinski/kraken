from __future__ import annotations

from datetime import UTC, datetime

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from kraken_core.frame_matrix.qt import FrameMatrixView
from kraken_core.frame_matrix.qt_parts.frame_rect import FrameRect
from kraken_core.frame_matrix.qt_parts.viewport_tile_item import FrameCellData, FrameSelection


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def test_border_and_fill_stay_stored_and_repeat_requests_do_not_recompute(qapp):
    view = FrameMatrixView(2, 1)
    view.set_visual_modes("quality", "status")
    view.set_cells(
        (
            FrameCellData(1, 1, payload={"quality": 0.0, "review_status": "not_checked"}),
            FrameCellData(2, 1, payload={"quality": 1.0, "review_status": "checked"}),
        )
    )
    calls = 0
    original = view._cell_visual_color

    def counting(cell, mode):
        nonlocal calls
        calls += 1
        return original(cell, mode)

    view._cell_visual_color = counting
    first = view.cell_visuals()
    second = view.cell_visuals()

    assert calls == 0
    assert first is second
    assert [(item.x, item.y) for item in first] == [(1, 1), (2, 1)]
    assert view.border_colors() is not view.fill_colors()
    assert view.border_colors()[(1, 1)].red() > 200
    assert view.fill_colors()[(1, 1)].name() == "#64748b"
    assert first[0].border is view.border_colors()[(1, 1)]
    assert first[0].fill is view.fill_colors()[(1, 1)]
    view.close()


def test_paint_reads_the_stored_visual_array(qapp, monkeypatch):
    view = FrameMatrixView(3, 2)
    view.resize(400, 300)
    view.show()
    view.set_cells(FrameCellData(x, y, status="image_ready") for y in range(1, 3) for x in range(1, 4))
    qapp.processEvents()

    def forbidden(cell, mode):
        raise AssertionError(f"visual color recomputed for {(cell.x, cell.y)} mode {mode}")

    monkeypatch.setattr(view, "_cell_visual_color", forbidden)
    view.viewport().update()
    qapp.processEvents()
    view.close()


def test_time_color_requests_reuse_the_stored_span(qapp):
    view = FrameMatrixView(2, 1)
    oldest = FrameCellData(1, 1, payload={"modified_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat()})
    newest = FrameCellData(2, 1, payload={"modified_at": datetime(2026, 1, 2, tzinfo=UTC).isoformat()})
    view.set_cells((oldest, newest))
    prepared = view._time_span_builds

    old_time = view._cell_visual_color(oldest, "time")
    new_time = view._cell_visual_color(newest, "time")

    assert view._time_span_builds == prepared
    assert old_time.red() > old_time.green()
    assert new_time.green() > new_time.red()
    view.close()


def test_time_color_stays_on_the_layer_scale_when_the_view_or_selection_narrows(qapp):
    view = FrameMatrixView(2, 1)
    view.set_visual_modes("time", "time")
    view.set_cells(
        (
            FrameCellData(1, 1, payload={"modified_at": 0}),
            FrameCellData(2, 1, payload={"modified_at": 100}),
        )
    )
    view.set_cells(
        (
            FrameCellData(1, 1, payload={"modified_at": 40}),
            FrameCellData(2, 1, payload={"modified_at": 60}),
        )
    )

    early = view.border_colors()[(1, 1)]
    late = view.fill_colors()[(2, 1)]
    assert view._time_span == (0.0, 100.0)
    assert early.red() < 220
    assert late.green() < 220
    assert early.red() > view.border_colors()[(2, 1)].red()

    border = view.border_colors()[(1, 1)].rgba()
    fill = view.fill_colors()[(1, 1)].rgba()
    view.set_selection(FrameRect(1, 1, 1, 1))
    assert view._time_span == (0.0, 100.0)
    assert view.border_colors()[(1, 1)].rgba() == border
    assert view.fill_colors()[(1, 1)].rgba() == fill
    view.close()


def test_none_border_mode_hides_the_cell_frame(qapp):
    view = FrameMatrixView(1, 1)
    view.set_visual_modes("none", "status")
    view.set_cells((FrameCellData(1, 1, payload={"review_status": "checked"}),))

    assert view.visual_modes() == ("none", "status")
    assert not view.border_colors()[(1, 1)].isValid()
    assert view.fill_colors()[(1, 1)].name() == "#22c55e"
    with pytest.raises(ValueError):
        view.set_visual_modes("none", "none")
    view.close()


def test_selection_frame_and_fill_are_one_stored_array(qapp, monkeypatch):
    view = FrameMatrixView(3, 2)
    view.resize(400, 300)
    view.show()
    view.set_selection(FrameRect(1, 1, 2, 1))
    qapp.processEvents()

    paints = view.selection_paints()
    assert [(round(item.fill.left()), round(item.fill.top())) for item in paints] == [(0, 0), (48, 0)]
    assert paints is view.selection_paints()
    assert view.selection_fill_rects() is not view.selection_border_rects()
    assert view.selection_fill_rects()[0] is paints[0].fill
    assert view.selection_border_rects()[0] is paints[0].border

    def forbidden(self, x, y):
        raise AssertionError(f"selection recalculated for {(x, y)}")

    monkeypatch.setattr(FrameSelection, "contains", forbidden)
    view.viewport().update()
    qapp.processEvents()
    view.close()
