from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from ...domain import PolygonData


@dataclass(frozen=True, slots=True)
class FrameLoadPayload:
    image_path: str
    source_image: object | None = None
    polygons: tuple[PolygonData, ...] = ()
    vectors_only: bool = False


class FrameLoadSignals(QObject):
    result = pyqtSignal(int, object)
    error = pyqtSignal(int, str, str)
    finished = pyqtSignal(int, str)


class FrameLoadRunnable(QRunnable):
    """Load frame pixels and/or vector overlay off the UI thread."""

    def __init__(
        self,
        request_id: int,
        image_path: str,
        *,
        load_source_image: Callable[[str], object] | None,
        load_cif_overlay: Callable[[str], list[PolygonData]],
        load_vectors: bool,
        vectors_only: bool,
    ) -> None:
        super().__init__()
        self.request_id = int(request_id)
        self.image_path = str(image_path)
        self._load_source = load_source_image
        self._load_cif = load_cif_overlay
        self.load_vectors = bool(load_vectors)
        self.vectors_only = bool(vectors_only)
        self.signals = FrameLoadSignals()

    def run(self) -> None:
        try:
            if self.vectors_only:
                polygons = tuple(self._load_cif(self.image_path))
                payload = FrameLoadPayload(
                    image_path=self.image_path,
                    polygons=polygons,
                    vectors_only=True,
                )
            else:
                source_image = None if self._load_source is None else self._load_source(self.image_path)
                polygons: tuple[PolygonData, ...] = ()
                if self.load_vectors:
                    polygons = tuple(self._load_cif(self.image_path))
                payload = FrameLoadPayload(
                    image_path=self.image_path,
                    source_image=source_image,
                    polygons=polygons,
                )
            try:
                self.signals.result.emit(self.request_id, payload)
            except RuntimeError:
                return
        except Exception as exc:
            try:
                self.signals.error.emit(self.request_id, self.image_path, str(exc))
            except RuntimeError:
                return
        finally:
            try:
                self.signals.finished.emit(self.request_id, self.image_path)
            except RuntimeError:
                return
