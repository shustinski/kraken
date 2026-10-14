"""qt: context menu event."""
from __future__ import annotations
from PyQt6.QtGui import QContextMenuEvent

from kraken_core.frame_matrix.qt_parts.viewport_tile_item import (
    FrameContext,
    FrameSelection,
)


class ContextMenuEventMixin:
    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # type: ignore[override]
        coordinate = self.frame_at_viewport_pos(event.pos())
        if coordinate is None:
            event.ignore()
            return
        x, y = coordinate
        if not self._selection.contains(x, y):
            self.set_selection(FrameSelection.single(x, y))
        context = FrameContext(x, y, self.cell_data(x, y), self._selection)
        self.contextMenuRequested.emit(context, event.globalPos())
        event.accept()
