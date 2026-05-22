from __future__ import annotations

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal
from PyQt6.QtGui import QImage

from .image_conversion import cv_to_qimage


class EditorDisplaySignals(QObject):
    result = pyqtSignal(int, str, object)
    finished = pyqtSignal(int, str)


class EditorDisplayRunnable(QRunnable):
    """Convert a frame buffer to QImage off the UI thread."""

    def __init__(self, request_id: int, image_path: str, image: object) -> None:
        super().__init__()
        self.request_id = int(request_id)
        self.image_path = str(image_path)
        self.image = image
        self.signals = EditorDisplaySignals()

    def run(self) -> None:
        qimage = QImage()
        try:
            if self.image is not None:
                qimage = cv_to_qimage(self.image)
        except Exception:
            qimage = QImage()
        try:
            self.signals.result.emit(self.request_id, self.image_path, qimage)
        except RuntimeError:
            return
        try:
            self.signals.finished.emit(self.request_id, self.image_path)
        except RuntimeError:
            return
