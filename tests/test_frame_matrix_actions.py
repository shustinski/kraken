from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QContextMenuEvent
from PyQt6.QtWidgets import QApplication

from kraken_core.frame_matrix.actions import FrameMatrixAction
from kraken_core.frame_matrix.qt import FrameContext, FrameMatrixView


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _boom(_context: FrameContext) -> bool:
    raise RuntimeError("predicate failed")


class _ActionMatrix(FrameMatrixView):
    chosen_id: str | None = None

    def frame_actions(self, context):
        return (
            FrameMatrixAction("first", "First"),
            FrameMatrixAction("hidden", "Hidden", lambda _context: False),
            FrameMatrixAction("broken", "Broken", _boom),
            FrameMatrixAction("titled", lambda context: f"Frame {context.x}"),
        )

    def _exec_frame_menu(self, menu, global_position):
        for action in menu.actions():
            if action.data() == self.chosen_id:
                return action
        return None


def _context_event(view: FrameMatrixView, x: int, y: int) -> QContextMenuEvent:
    position = view.mapFromScene(view.scene_rect_for_frame(x, y).center())
    return QContextMenuEvent(QContextMenuEvent.Reason.Mouse, position, view.mapToGlobal(position))


def test_visible_actions_skip_hidden_and_failing_predicates(qapp):
    view = _ActionMatrix(3, 1)
    context = FrameContext(2, 1, view.cell_data(2, 1), view.selection())

    visible = view.visible_frame_actions(context)

    assert [(action.id, title) for action, title in visible] == [("first", "First"), ("titled", "Frame 2")]


def test_chosen_action_is_reported_with_context(qapp):
    view = _ActionMatrix(3, 1)
    view.resize(300, 120)
    view.show()
    qapp.processEvents()
    view.chosen_id = "first"
    emitted = []
    view.frameActionTriggered.connect(lambda action_id, context: emitted.append((action_id, context)))

    view.contextMenuEvent(_context_event(view, 2, 1))

    assert len(emitted) == 1
    action_id, context = emitted[0]
    assert action_id == "first"
    assert (context.x, context.y) == (2, 1)
    assert context.selection.contains(2, 1)


def test_matrix_without_actions_keeps_context_menu_signal(qapp):
    view = FrameMatrixView(3, 1)
    view.resize(300, 120)
    view.show()
    qapp.processEvents()
    requested = []
    view.contextMenuRequested.connect(lambda context, position: requested.append(context))

    view.contextMenuEvent(_context_event(view, 1, 1))

    assert [(context.x, context.y) for context in requested] == [(1, 1)]
