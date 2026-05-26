from __future__ import annotations

import json
import numpy as np
import subprocess
import sys
import tempfile
from PyQt6.QtCore import QObject, QRunnable, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QImage

from ...application.frame_lod import PyramidFrameStore, ZarrFrameStore


class PyramidFrameLoadSignals(QObject):
    result = pyqtSignal(int, int, int, object)
    error = pyqtSignal(int, int, int, str)


class PyramidThumbnailLoadSignals(QObject):
    result = pyqtSignal(int, int, int, int, int, object)
    error = pyqtSignal(int, int, int, int, int, str)


class ZarrPyramidBuildSignals(QObject):
    finished = pyqtSignal(int, object)
    error = pyqtSignal(int, str)


class ZarrPyramidBuildRunnable(QRunnable):
    def __init__(self, generation: int, store: ZarrFrameStore) -> None:
        super().__init__()
        self.generation = int(generation)
        self.store = store
        self.signals = ZarrPyramidBuildSignals()

    def run(self) -> None:
        manifest_path = None
        try:
            if self.store.zarr_path is None:
                raise RuntimeError("Zarr output path is not configured.")
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as manifest:
                manifest_path = manifest.name
                json.dump(
                    {
                        "zarr_path": str(self.store.zarr_path),
                        "image_paths": [str(path) for path in self.store.image_paths],
                    },
                    manifest,
                )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "contour.adapters.qt.zarr_build_worker",
                    manifest_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                message = (completed.stderr or completed.stdout or "Failed to build Zarr pyramid.").strip()
                raise RuntimeError(message)
            self.signals.finished.emit(self.generation, self.store)
        except Exception as exc:
            self.signals.error.emit(self.generation, str(exc))
        finally:
            if manifest_path:
                try:
                    from pathlib import Path

                    Path(manifest_path).unlink(missing_ok=True)
                except Exception:
                    pass


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
            self.signals.result.emit(self.generation, self.frame_id, self.lod, qimage)
        except Exception as exc:
            self.signals.error.emit(self.generation, self.frame_id, self.lod, str(exc))


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
            self.signals.result.emit(
                self.generation,
                self.frame_id,
                self.lod,
                self.target_width,
                self.target_height,
                qimage,
            )
        except Exception as exc:
            self.signals.error.emit(
                self.generation,
                self.frame_id,
                self.lod,
                self.target_width,
                self.target_height,
                str(exc),
            )


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
