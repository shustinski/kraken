from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal
from PyQt6.QtGui import QImage

from .image_conversion import cv_to_qimage

_LOGGER = logging.getLogger(__name__)


class EditorDisplaySignals(QObject):
    result = pyqtSignal(int, str, object)
    finished = pyqtSignal(int, str)


class EditorDisplayRunnable(QRunnable):
    """Convert a frame buffer to QImage off the UI thread."""

    def __init__(
        self,
        request_id: int,
        image_path: str,
        image: object,
        *,
        profile_session: object | None = None,
    ) -> None:
        super().__init__()
        self.request_id = int(request_id)
        self.image_path = str(image_path)
        self.image = image
        self._profile_session = profile_session
        self.signals = EditorDisplaySignals()

    def run(self) -> None:
        from ...infrastructure.frame_switch_profiler import profile_callable

        def _convert() -> QImage:
            if self.image is None:
                return QImage()
            return cv_to_qimage(self.image)

        qimage = QImage()
        try:
            qimage = profile_callable("editor_display", self._profile_session, _convert)
        except Exception:
            _LOGGER.exception("Editor image conversion failed for %s", self.image_path)
            qimage = QImage()
        try:
            self.signals.result.emit(self.request_id, self.image_path, qimage)
        except RuntimeError:
            return
        try:
            self.signals.finished.emit(self.request_id, self.image_path)
        except RuntimeError:
            return
