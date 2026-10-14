"""processing_mixin: reload current frame vectors."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
)


class ReloadCurrentFrameVectorsMixin:
    def _reload_current_frame_vectors(self: Any) -> None:
        current = self._workspace.current_image_path
        if not current:
            return
        normalized = str(Path(current))
        state = self._workspace.current_state
        if state is None or state.source_image is None:
            self.load_image(normalized, load_vectors=True)
            return
        running_path = getattr(self, "_frame_load_running_path", None)
        if running_path is not None or getattr(self, "_loading_image_path", None) is not None:
            self._mark_thumbnail_grid_rebuild_pending()
            self._frame_load_pending = (normalized, True, False)
            return
        self._loading_image_path = normalized
        try:
            self._begin_frame_vectors_reload(normalized)
        except Exception:
            if getattr(self, "_loading_image_path", None) == normalized:
                self._loading_image_path = None
            raise
