from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from ...application.polygon_antialiasing import antialias_polygons
from ...domain import PolygonData
from ...i18n import tr
from ...serializers import load_polygons_vector, save_polygons_vector


@dataclass(frozen=True, slots=True)
class AntialiasCifWorkItem:
    stem: str
    cif_path: str
    image_path: str | None
    polygons: tuple[PolygonData, ...] | None
    image_size: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class AntialiasCifItemResult:
    run_id: int
    stem: str
    cif_path: str
    image_path: str | None
    polygons: tuple[PolygonData, ...]
    image_size: tuple[int, int] | None
    changed: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AntialiasCifJobSummary:
    run_id: int
    saved_count: int
    changed_count: int
    failed: tuple[str, ...] = field(default_factory=tuple)
    cancelled: bool = False
    grade: int = 1


class AntialiasCifSignals(QObject):
    progress = pyqtSignal(int, int, int)
    item_finished = pyqtSignal(object)
    finished = pyqtSignal(object)


class AntialiasCifRunnable(QRunnable):
    def __init__(
        self,
        *,
        items: tuple[AntialiasCifWorkItem, ...],
        grade: int,
        signals: AntialiasCifSignals,
        cancel_event: threading.Event,
        run_id: int,
    ) -> None:
        super().__init__()
        self._items = items
        self._grade = int(grade)
        self._signals = signals
        self._cancel_event = cancel_event
        self._run_id = int(run_id)

    def run(self) -> None:
        total = len(self._items)
        saved_count = 0
        changed_count = 0
        failed: list[str] = []
        cancelled = False
        self._signals.progress.emit(self._run_id, 0, total)
        try:
            for index, item in enumerate(self._items, start=1):
                if self._cancel_event.is_set():
                    cancelled = True
                    break
                try:
                    result = _antialias_work_item(item, self._grade, self._run_id)
                except Exception as exc:
                    failed.append(f"{Path(item.cif_path).name}: {exc}")
                    result = AntialiasCifItemResult(
                        run_id=self._run_id,
                        stem=item.stem,
                        cif_path=item.cif_path,
                        image_path=item.image_path,
                        polygons=(),
                        image_size=item.image_size,
                        changed=False,
                        error=str(exc),
                    )
                else:
                    if result.changed:
                        changed_count += 1
                        saved_count += 1
                if result.changed or result.error:
                    self._signals.item_finished.emit(result)
                self._signals.progress.emit(self._run_id, index, total)
            if self._cancel_event.is_set():
                cancelled = True
        finally:
            self._signals.finished.emit(
                AntialiasCifJobSummary(
                    run_id=self._run_id,
                    saved_count=saved_count,
                    changed_count=changed_count,
                    failed=tuple(failed),
                    cancelled=cancelled,
                    grade=self._grade,
                )
            )


def _antialias_work_item(item: AntialiasCifWorkItem, grade: int, run_id: int) -> AntialiasCifItemResult:
    image_path = item.image_path
    image_size = item.image_size
    if item.polygons is not None and image_size is not None:
        polygons = [polygon.clone() for polygon in item.polygons]
    else:
        loaded_name, loaded_size, loaded_polygons = load_polygons_vector(item.cif_path)
        polygons = (
            [polygon.clone() for polygon in item.polygons]
            if item.polygons is not None
            else loaded_polygons
        )
        if image_size is None:
            image_size = loaded_size
        if image_path is None and loaded_name:
            image_path = loaded_name
    if image_size is None:
        raise ValueError(tr("cif_size_header_missing", path=item.cif_path))
    if image_path is None:
        image_path = str(Path(item.cif_path).with_suffix(""))
    antialiased, changed = antialias_polygons(polygons, grade)
    if changed:
        save_polygons_vector(item.cif_path, image_path, antialiased, image_size=image_size)
    return AntialiasCifItemResult(
        run_id=run_id,
        stem=item.stem,
        cif_path=item.cif_path,
        image_path=image_path,
        polygons=tuple(polygon.clone() for polygon in antialiased),
        image_size=image_size,
        changed=changed,
    )
