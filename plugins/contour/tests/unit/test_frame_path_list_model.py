from __future__ import annotations

from types import SimpleNamespace

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from contour.ui.frame_path_list_model import FramePathListModel
from contour.ui.item_status_painting import FRAME_STATUS_ROLE


def test_frame_path_list_model_computes_paint_roles_once_per_row() -> None:
    calls: list[str] = []

    def _row_paint_roles(path: str) -> dict[int, object]:
        calls.append(path)
        return {
            FRAME_STATUS_ROLE: "ok",
            int(Qt.ItemDataRole.BackgroundRole): QColor("#112233"),
            int(Qt.ItemDataRole.ForegroundRole): QColor("#445566"),
        }

    widget = SimpleNamespace(
        _ui_language="en",
        _image_list_model_row_paint_roles=_row_paint_roles,
    )
    model = FramePathListModel(None)
    model._widget = widget
    model.set_paths([r"d:\frames\a.png", r"d:\frames\b.png"])

    index = model.index(0, 0)
    assert model.data(index, Qt.ItemDataRole.DisplayRole) == "a"
    assert model.data(index, Qt.ItemDataRole.UserRole) == r"d:\frames\a.png"
    assert model.data(index, Qt.ItemDataRole.ToolTipRole) == r"File path: d:\frames\a.png"
    assert model.data(index, FRAME_STATUS_ROLE) == "ok"
    assert model.data(index, Qt.ItemDataRole.BackgroundRole).name() == "#112233"
    assert model.data(index, Qt.ItemDataRole.ForegroundRole).name() == "#445566"
    assert model.data(index, FRAME_STATUS_ROLE) == "ok"
    assert calls == [r"d:\frames\a.png"]

    model.invalidate_path(r"d:\frames\a.png")
    assert model.data(index, FRAME_STATUS_ROLE) == "ok"
    assert calls == [r"d:\frames\a.png", r"d:\frames\a.png"]


def test_frame_path_list_model_invalidate_row_range_only_touches_span() -> None:
    calls: list[str] = []

    def _row_paint_roles(path: str) -> dict[int, object]:
        calls.append(path)
        return {FRAME_STATUS_ROLE: "ok"}

    model = FramePathListModel(None)
    model._widget = SimpleNamespace(_image_list_model_row_paint_roles=_row_paint_roles)
    model.set_paths([f"f{i}.png" for i in range(5)])
    for row in range(5):
        assert model.data(model.index(row, 0), FRAME_STATUS_ROLE) == "ok"
    calls.clear()

    model.invalidate_row_range(1, 2)
    assert model.data(model.index(0, 0), FRAME_STATUS_ROLE) == "ok"
    assert model.data(model.index(1, 0), FRAME_STATUS_ROLE) == "ok"
    assert model.data(model.index(2, 0), FRAME_STATUS_ROLE) == "ok"
    assert calls == ["f1.png", "f2.png"]
