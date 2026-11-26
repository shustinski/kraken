"""navigation_mixin: scroll thumbnail grid row."""
from __future__ import annotations
from typing import Any


class ScrollThumbnailGridRowMixin:
    def _scroll_thumbnail_grid_to_row(self: Any, row: int) -> None:
        if not self._frame_matrix_enabled() or not hasattr(self, "thumbnail_grid") or not hasattr(self, "thumbnail_grid_scroll_area"):
            return
        if row < 0 or row >= self.thumbnail_grid.count():
            return
        self.thumbnail_grid.ensureRowVisible(row)
        self._schedule_visible_thumbnail_loads()
