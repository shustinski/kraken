"""qt: context menu event."""
from __future__ import annotations
from collections.abc import Sequence
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QAction, QContextMenuEvent
from PyQt6.QtWidgets import QMenu

from kraken_core.frame_matrix.actions import FrameMatrixAction
from kraken_core.frame_matrix.qt_parts.module_state import _LOGGER
from kraken_core.frame_matrix.qt_parts.viewport_tile_item import (
    FrameContext,
    FrameSelection,
)


class ContextMenuEventMixin:
    def frame_actions(self, context: FrameContext) -> Sequence[FrameMatrixAction]:
        """Actions this matrix offers for ``context``. Subclasses declare their own."""

        del context
        return ()

    def visible_frame_actions(self, context: FrameContext) -> tuple[tuple[FrameMatrixAction, str], ...]:
        visible: list[tuple[FrameMatrixAction, str]] = []
        for action in self.frame_actions(context):
            try:
                if action.visible_for(context):
                    visible.append((action, action.title_for(context)))
            except Exception:
                _LOGGER.exception("Frame action %r failed to evaluate", action.id)
        return tuple(visible)

    def _exec_frame_menu(self, menu: QMenu, global_position: QPoint) -> QAction | None:
        return menu.exec(global_position)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # type: ignore[override]
        coordinate = self.frame_at_viewport_pos(event.pos())
        if coordinate is None:
            event.ignore()
            return
        x, y = coordinate
        if not self._selection.contains(x, y):
            self.set_selection(FrameSelection.single(x, y))
        context = FrameContext(x, y, self.cell_data(x, y), self._selection)
        event.accept()
        if not tuple(self.frame_actions(context)):
            self.contextMenuRequested.emit(context, event.globalPos())
            return
        visible = self.visible_frame_actions(context)
        if not visible:
            return
        menu = QMenu(self)
        for action, title in visible:
            menu.addAction(title).setData(action.id)
        chosen = self._exec_frame_menu(menu, event.globalPos())
        menu.deleteLater()
        if chosen is not None and chosen.data():
            self.frameActionTriggered.emit(str(chosen.data()), context)
