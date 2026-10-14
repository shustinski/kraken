"""processing_mixin: drain pending frame load."""
from __future__ import annotations
from typing import Any


class DrainPendingFrameLoadMixin:
    def _drain_pending_frame_load(self: Any) -> None:
        pending = getattr(self, "_frame_load_pending", None)
        if pending:
            self._frame_load_pending = None
            path, load_vectors, preserve_editor_view_position = pending
            self.load_image(
                path,
                load_vectors=load_vectors,
                preserve_editor_view_position=preserve_editor_view_position,
            )
            return
        self._flush_pending_thumbnail_grid_rebuild()
        self._schedule_frame_prefetch()
