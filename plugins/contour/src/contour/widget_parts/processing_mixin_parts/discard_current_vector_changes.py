"""processing_mixin: discard current vector changes."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
)


class DiscardCurrentVectorChangesMixin:
    def _discard_current_vector_changes(self: Any) -> None:
        state = self._workspace.current_state
        path = self._workspace.current_image_path
        if state is None or path is None:
            return
        restored = [polygon.clone() for polygon in state.reference_polygons]
        self._updating_views = True
        try:
            state.polygons = restored
            self._workspace.update_current_polygons(restored)
            self.polygon_editor.set_polygons(restored)
        finally:
            self._updating_views = False
        self._persisted_highlight_paths.discard(str(Path(path)))
        self._update_frame_item_status(path)
        self._update_vector_edit_status_label()
