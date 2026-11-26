from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from kraken_core.frame_matrix.qt import FrameContext, FrameMatrixWidget, FrameRect, FrameSelection
from kraken_hub.presentation.qt.layer_matrix import LayerFrameMatrixWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _Host:
    def __init__(self, *, assign: bool = False, vector_file: object | None = None) -> None:
        self.assign = assign
        self.vector_file = vector_file

    def can_assign_work(self) -> bool:
        return self.assign

    def vector_file_for(self, x: int, y: int) -> object | None:
        return self.vector_file


def _visible(view: LayerFrameMatrixWidget, selection: FrameSelection) -> list[tuple[str, str]]:
    context = FrameContext(1, 1, view.cell_data(1, 1), selection)
    return [(action.id, title) for action, title in view.visible_frame_actions(context)]


def test_layer_matrix_shares_the_kraken_matrix_class(qapp):
    assert issubclass(LayerFrameMatrixWidget, FrameMatrixWidget)


def test_single_frame_actions(qapp):
    view = LayerFrameMatrixWidget(4, 4)
    view.set_action_host(_Host())

    assert _visible(view, FrameSelection.single(1, 1)) == [
        ("export_files", "Выгрузить файлы кадра…"),
        ("frame_properties", "Свойства кадра…"),
    ]


def test_multi_frame_selection_hides_frame_properties(qapp):
    view = LayerFrameMatrixWidget(4, 4)
    view.set_action_host(_Host(assign=True, vector_file=object()))

    assert _visible(view, FrameSelection((FrameRect(1, 1, 3, 1),))) == [
        ("export_files", "Выгрузить файлы выбранных кадров (3)…"),
        ("assign_work", "Выдать работу…"),
    ]


def test_assign_work_and_vector_properties_follow_host(qapp):
    view = LayerFrameMatrixWidget(4, 4)
    view.set_action_host(_Host(assign=True, vector_file=object()))

    assert [action_id for action_id, _title in _visible(view, FrameSelection.single(1, 1))] == [
        "export_files",
        "assign_work",
        "vector_properties",
        "frame_properties",
    ]
