"""processing_mixin: load image."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
)


class LoadImageMixin:
    def load_image(
        self: Any,
        path: str,
        *,
        load_vectors: bool | None = None,
        preserve_editor_view_position: bool = False,
    ) -> None:
        normalized_load_path = str(Path(path))
        self._preserve_editor_view_position_path = (
            normalized_load_path if preserve_editor_view_position else None
        )
        if load_vectors is None:
            load_vectors = not self._should_defer_vector_load()
        if self._image_path_in_image_list(normalized_load_path):
            self._desired_image_path = normalized_load_path
        active_load_path = getattr(self, "_loading_image_path", None)
        if active_load_path is not None:
            if active_load_path == normalized_load_path:
                if bool(load_vectors):
                    pending = getattr(self, "_frame_load_pending", None)
                    if pending is None or str(Path(pending[0])) == normalized_load_path:
                        preserve_pending = bool(pending[2]) if pending is not None else False
                        self._frame_load_pending = (
                            normalized_load_path,
                            True,
                            bool(preserve_editor_view_position) or preserve_pending,
                        )
                return
            self._frame_load_pending = (
                normalized_load_path,
                bool(load_vectors),
                bool(preserve_editor_view_position),
            )
            return
        running_path = getattr(self, "_frame_load_running_path", None)
        if running_path is not None:
            self._frame_load_pending = (
                normalized_load_path,
                bool(load_vectors),
                bool(preserve_editor_view_position),
            )
            return
        self._loading_image_path = normalized_load_path
        if hasattr(self, "_pause_thumbnail_radial_fill"):
            self._pause_thumbnail_radial_fill()
        try:
            self._begin_frame_load(normalized_load_path, load_vectors=bool(load_vectors))
        except Exception:
            if getattr(self, "_loading_image_path", None) == normalized_load_path:
                self._loading_image_path = None
            raise
