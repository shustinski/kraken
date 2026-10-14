"""navigation_mixin: scroll thumbnail grid row."""
from __future__ import annotations
from typing import Any


class ScrollThumbnailGridRowMixin:
    def _scroll_thumbnail_grid_to_row(self: Any, row: int) -> None:
        if not self._frame_matrix_enabled() or not hasattr(self, "thumbnail_grid") or not hasattr(self, "thumbnail_grid_scroll_area"):
            return
        if row < 0 or row >= self.thumbnail_grid.count():
            return
        item = self.thumbnail_grid.item(row)
        if item is None:
            return
        rect = self.thumbnail_grid.visualItemRect(item)
        if rect.isNull():
            return
        self.thumbnail_grid_scroll_area.ensureVisible(rect.center().x(), rect.center().y(), 0, 0)
        viewport = self.thumbnail_grid_scroll_area.viewport().rect()
        horizontal = self.thumbnail_grid_scroll_area.horizontalScrollBar()
        vertical = self.thumbnail_grid_scroll_area.verticalScrollBar()
        margin = 0
        if rect.left() < horizontal.value():
            horizontal.setValue(max(horizontal.minimum(), rect.left() - margin))
        elif rect.right() > horizontal.value() + viewport.width():
            horizontal.setValue(min(horizontal.maximum(), rect.right() - viewport.width() + margin))
        if rect.top() < vertical.value():
            vertical.setValue(max(vertical.minimum(), rect.top() - margin))
        elif rect.bottom() > vertical.value() + viewport.height():
            vertical.setValue(min(vertical.maximum(), rect.bottom() - viewport.height() + margin))
        self._schedule_visible_thumbnail_loads()
