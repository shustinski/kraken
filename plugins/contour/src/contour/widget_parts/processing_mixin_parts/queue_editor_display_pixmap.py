"""processing_mixin: queue editor display pixmap."""
from __future__ import annotations
from typing import Any
from PyQt6.QtGui import QImage
from ...adapters.qt.object_validity import qt_object_is_valid
from contour.widget_parts._imports import (
    EditorDisplayRunnable,
    Path,
    QPixmap,
    np,
)


class QueueEditorDisplayPixmapMixin:
    def _queue_editor_display_pixmap(
        self: Any,
        image_path: str,
        display_image: np.ndarray | None,
        *,
        cache_key: tuple[str, str, str],
        preserve_view: bool = False,
        cache_only: bool = False,
    ) -> None:
        if display_image is None:
            self.polygon_editor.set_image_pixmap(QPixmap(), preserve_view=preserve_view)
            return
        session = self._frame_switch_profile_for_path(image_path)
        if session is not None:
            session.mark_pending("editor_display")
        if cache_only:
            request_id = 0
        else:
            self._editor_display_request_serial = int(getattr(self, "_editor_display_request_serial", 0)) + 1
            request_id = self._editor_display_request_serial
        target_path = str(Path(image_path))
        runnable = EditorDisplayRunnable(
            request_id,
            target_path,
            display_image,
            profile_session=session,
        )

        def _on_display_ready(req_id: int, path: str, qimage: object) -> None:
            if cache_only:
                getattr(self, "_editor_source_cache_pending_keys", set()).discard(cache_key)
            if bool(getattr(self, "_closing", False)) or not qt_object_is_valid(self):
                return
            editor = getattr(self, "polygon_editor", None)
            if not qt_object_is_valid(editor):
                return
            pixmap = QPixmap()
            if isinstance(qimage, QImage):
                try:
                    pixmap = QPixmap.fromImage(qimage)
                except Exception:
                    pixmap = QPixmap()
            current_path = str(Path(self._workspace.current_image_path or ""))
            target_path = str(Path(path))
            same_frame_state = target_path == current_path
            is_current_display = (
                not cache_only
                and req_id == self._editor_display_request_serial
                and same_frame_state
                and (
                    self._workspace.current_state is None
                    or self._editor_display_cache_key(path, self._workspace.current_state) == cache_key
                )
            )
            if pixmap.isNull():
                if is_current_display:
                    def _clear_null() -> None:
                        self.polygon_editor.set_image_pixmap(QPixmap(), preserve_view=preserve_view)
                        self._pending_editor_frame_apply = None

                    self._suspend_editor_viewport_updates(_clear_null)
                return
            if not same_frame_state:
                return
            # A newer filtered display request may supersede the initial
            # source-frame request before this queued signal is delivered.
            # Keep the completed pixmap for source preview, but
            # only the newest matching request may alter the visible image.
            cache = getattr(self, "_editor_pixmap_cache", {})
            cache[cache_key] = pixmap
            while len(cache) > 48:
                cache.pop(next(iter(cache)))
            self._editor_pixmap_cache = cache
            if not is_current_display:
                return

            def _commit_display() -> None:
                self.polygon_editor.set_image_pixmap(pixmap, preserve_view=preserve_view)
                self._last_editor_display_cache_key = cache_key
                self._last_editor_display_path = str(Path(path))
                session = self._frame_switch_profile_for_path(path)
                if session is not None:
                    session.complete_pending("editor_display", suffix="_ready")
                self._flush_pending_editor_frame_apply(str(Path(path)))

            self._suspend_editor_viewport_updates(_commit_display)

        runnable.signals.result.connect(_on_display_ready)
        self._editor_display_thread_pool.start(runnable)
