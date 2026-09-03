from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QObject, QRunnable, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QImage

from ...application.frame_lod import PyramidFrameStore

_LOGGER = logging.getLogger(__name__)


class PyramidFrameLoadSignals(QObject):
    result = pyqtSignal(int, int, int, object)
    error = pyqtSignal(int, int, int, str)


class PyramidThumbnailLoadSignals(QObject):
    result = pyqtSignal(int, int, int, int, int, object)
    error = pyqtSignal(int, int, int, int, int, str)


def _emit_if_alive(signal_owner: QObject, signal_name: str, *args: object) -> None:
    """Treat deletion of a runnable's Qt signal owner as normal cancellation."""

    try:
        getattr(signal_owner, signal_name).emit(*args)
    except RuntimeError:
        # The view can be rebuilt or closed while OpenCV is still decoding.
        # In that case Qt deletes the signal object before the worker returns.
        return


class PyramidFrameLoadRunnable(QRunnable):
    def __init__(self, generation: int, frame_id: int, lod: int, store: PyramidFrameStore) -> None:
        super().__init__()
        self.generation = int(generation)
        self.frame_id = int(frame_id)
        self.lod = int(lod)
        self.store = store
        self.signals = PyramidFrameLoadSignals()

    def run(self) -> None:
        try:
            array = self.store.get_frame(self.frame_id, self.lod)
            qimage = qimage_from_array(array)
        except Exception as exc:
            _LOGGER.exception("Pyramid frame loading failed for frame=%s lod=%s", self.frame_id, self.lod)
            _emit_if_alive(self.signals, "error", self.generation, self.frame_id, self.lod, str(exc))
            return
        _emit_if_alive(self.signals, "result", self.generation, self.frame_id, self.lod, qimage)


class PyramidThumbnailLoadRunnable(QRunnable):
    def __init__(
        self,
        generation: int,
        frame_id: int,
        lod: int,
        store: PyramidFrameStore,
        target_width: int,
        target_height: int,
    ) -> None:
        super().__init__()
        self.generation = int(generation)
        self.frame_id = int(frame_id)
        self.lod = int(lod)
        self.store = store
        self.target_width = max(1, int(target_width))
        self.target_height = max(1, int(target_height))
        self.signals = PyramidThumbnailLoadSignals()

    def run(self) -> None:
        try:
            array = self.store.get_thumbnail(
                self.frame_id,
                self.lod,
                max_size=max(self.target_width, self.target_height),
            )
            qimage = qimage_from_array(array)
            qimage = qimage.scaled(
                QSize(self.target_width, self.target_height),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            result_args = (
                self.generation,
                self.frame_id,
                self.lod,
                self.target_width,
                self.target_height,
                qimage,
            )
        except Exception as exc:
            _LOGGER.exception("Pyramid thumbnail loading failed for frame=%s lod=%s", self.frame_id, self.lod)
            _emit_if_alive(
                self.signals,
                "error",
                self.generation,
                self.frame_id,
                self.lod,
                self.target_width,
                self.target_height,
                str(exc),
            )
            return
        _emit_if_alive(self.signals, "result", *result_args)


def qimage_from_array(array: np.ndarray) -> QImage:
    image = np.ascontiguousarray(array)
    if image.ndim == 2:
        height, width = image.shape
        qimage = QImage(image.data, width, height, width, QImage.Format.Format_Grayscale8)
        return qimage.copy()
    if image.ndim == 3 and image.shape[2] == 3:
        height, width, _channels = image.shape
        bytes_per_line = 3 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return qimage.copy()
    if image.ndim == 3 and image.shape[2] == 4:
        height, width, _channels = image.shape
        bytes_per_line = 4 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
        return qimage.copy()
    squeezed = np.squeeze(image)
    if squeezed.ndim != image.ndim:
        return qimage_from_array(squeezed)
    raise ValueError(f"Unsupported frame array shape: {tuple(image.shape)}")
