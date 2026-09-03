from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from ...domain import PolygonData

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FrameLoadPayload:
    image_path: str
    source_image: object | None = None
    polygons: tuple[PolygonData, ...] = ()
    vectors_only: bool = False
    repair_reasons: dict[int, tuple[str, ...]] | None = None


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
        scan_repair: Callable[[list[PolygonData]], Mapping[int, list[str]]] | None = None,
    ) -> None:
        super().__init__()
        self.request_id = int(request_id)
        self.image_path = str(image_path)
        self._load_source = load_source_image
        self._load_cif = load_cif_overlay
        self._scan_repair = scan_repair
        self.load_vectors = bool(load_vectors)
        self.vectors_only = bool(vectors_only)
        self.signals = FrameLoadSignals()

    def _repair_payload(self, polygons: list[PolygonData]) -> dict[int, tuple[str, ...]] | None:
        if self._scan_repair is None or not polygons:
            return None
        reasons = self._scan_repair(polygons)
        return {int(polygon_id): tuple(codes) for polygon_id, codes in reasons.items()}

    def run(self) -> None:
        try:
            if self.vectors_only:
                polygons_list = list(self._load_cif(self.image_path))
                polygons = tuple(polygons_list)
                payload = FrameLoadPayload(
                    image_path=self.image_path,
                    polygons=polygons,
                    vectors_only=True,
                    repair_reasons=self._repair_payload(polygons_list),
                )
            else:
                source_image = None if self._load_source is None else self._load_source(self.image_path)
                polygons_list: list[PolygonData] = []
                if self.load_vectors:
                    polygons_list = list(self._load_cif(self.image_path))
                polygons = tuple(polygons_list)
                payload = FrameLoadPayload(
                    image_path=self.image_path,
                    source_image=source_image,
                    polygons=polygons,
                    repair_reasons=self._repair_payload(polygons_list),
                )
            try:
                self.signals.result.emit(self.request_id, payload)
            except RuntimeError:
                return
        except Exception as exc:
            _LOGGER.exception("Frame loading failed for %s", self.image_path)
            try:
                self.signals.error.emit(self.request_id, self.image_path, str(exc))
            except RuntimeError:
                return
        finally:
            try:
                self.signals.finished.emit(self.request_id, self.image_path)
            except RuntimeError:
                return


class GeometryValidationSignals(QObject):
    result = pyqtSignal(int, str, object)
    error = pyqtSignal(int, str, str)


class GeometryValidationRunnable(QRunnable):
    """Scan polygon repair reasons off the UI thread after a frame is shown."""

    def __init__(
        self,
        request_id: int,
        image_path: str,
        polygons: list[PolygonData],
        scan_repair: Callable[[list[PolygonData]], Mapping[int, list[str]]],
        *,
        profile_session: object | None = None,
    ) -> None:
        super().__init__()
        self.request_id = int(request_id)
        self.image_path = str(image_path)
        self._polygons = [polygon.clone() for polygon in polygons]
        self._scan_repair = scan_repair
        self._profile_session = profile_session
        self.signals = GeometryValidationSignals()

    def run(self) -> None:
        from ...infrastructure.frame_switch_profiler import profile_callable

        def _scan() -> dict[int, list[str]]:
            return {
                int(polygon_id): list(codes)
                for polygon_id, codes in self._scan_repair(self._polygons).items()
            }

        try:
            reasons = profile_callable("geometry_validation", self._profile_session, _scan)
            try:
                self.signals.result.emit(self.request_id, self.image_path, reasons)
            except RuntimeError:
                return
        except Exception as exc:
            _LOGGER.exception("Geometry validation failed for %s", self.image_path)
            try:
                self.signals.error.emit(self.request_id, self.image_path, str(exc))
            except RuntimeError:
                return
