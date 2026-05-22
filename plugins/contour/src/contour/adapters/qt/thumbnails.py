from __future__ import annotations

from PyQt6.QtCore import QObject, Qt, QRunnable, pyqtSignal
from PyQt6.QtGui import QImage

from ...utils import load_image_color_thumbnail
from .image_conversion import cv_to_qimage


class ThumbnailLoadSignals(QObject):
    result = pyqtSignal(int, str, object)
    finished = pyqtSignal(int, str)


class ThumbnailLoadRunnable(QRunnable):
    def __init__(self, generation: int, path: str, width: int, height: int) -> None:
        super().__init__()
        self.generation = int(generation)
        self.path = str(path)
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.signals = ThumbnailLoadSignals()

    def run(self) -> None:
        qimage: QImage | None = None
        try:
            image = load_image_color_thumbnail(self.path, self.width, self.height)
            qimage = cv_to_qimage(image)
            if not qimage.isNull():
                qimage = qimage.scaled(
                    self.width,
                    self.height,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
        except Exception:
            qimage = None
        try:
            self.signals.result.emit(self.generation, self.path, qimage)
        except RuntimeError:
            return
        try:
            self.signals.finished.emit(self.generation, self.path)
        except RuntimeError:
            return
