from __future__ import annotations

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from ...utils import load_image_color_thumbnail


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
        image = None
        try:
            image = load_image_color_thumbnail(self.path, self.width, self.height)
        except Exception:
            image = None
        # Widget shutdown can race with background thumbnail workers:
        # in that case underlying QObject wrappers may already be deleted.
        try:
            self.signals.result.emit(self.generation, self.path, image)
        except RuntimeError:
            return
        try:
            self.signals.finished.emit(self.generation, self.path)
        except RuntimeError:
            return
