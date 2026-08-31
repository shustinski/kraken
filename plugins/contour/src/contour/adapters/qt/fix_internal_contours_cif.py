from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from ...application.fix_internal_contours import (
    InternalContourFixStats,
    fix_internal_contour_display,
)
from ...domain import PolygonData
from ...i18n import tr
from ...serializers import load_polygons_vector, save_polygons_vector


@dataclass(frozen=True, slots=True)
class FixInternalContoursCifWorkItem:
    stem: str
    cif_path: str
    image_path: str | None
    polygons: tuple[PolygonData, ...] | None
    image_size: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class FixInternalContoursCifItemResult:
    run_id: int
    stem: str
    cif_path: str
    image_path: str | None
    polygons: tuple[PolygonData, ...]
    image_size: tuple[int, int] | None
    changed: bool
    fixed_families: int = 0
    fixed_hole_regions: int = 0
    checked_families: int = 0
    skipped_klayout_keyholes: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FixInternalContoursCifJobSummary:
    run_id: int
    stats: InternalContourFixStats
    saved_count: int = 0


class FixInternalContoursCifSignals(QObject):
    progress = pyqtSignal(int, int, int)
    item_finished = pyqtSignal(object)
    finished = pyqtSignal(object)


def run_fix_internal_contours_batch(
    items: tuple[FixInternalContoursCifWorkItem, ...],
    *,
    run_id: int,
    cancel_event: threading.Event | None = None,
) -> FixInternalContoursCifJobSummary:
    """Process CIF items on the Qt GUI thread (requires QPainter for analysis)."""

    checked_cif_files = 0
    checked_families = 0
    fixed_cif_files = 0
    fixed_families = 0
    fixed_hole_regions = 0
    skipped_klayout_keyholes = 0
    unchanged_cif_files = 0
    saved_count = 0
    failed: list[str] = []
    cancelled = False
    for item in items:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        try:
            result = _fix_internal_contours_work_item(item, run_id)
        except Exception as exc:
            failed.append(f"{Path(item.cif_path).name}: {exc}")
            result = FixInternalContoursCifItemResult(
                run_id=run_id,
                stem=item.stem,
                cif_path=item.cif_path,
                image_path=item.image_path,
                polygons=(),
                image_size=item.image_size,
                changed=False,
                error=str(exc),
            )
        checked_cif_files += 1
        checked_families += result.checked_families
        skipped_klayout_keyholes += result.skipped_klayout_keyholes
        if result.changed:
            fixed_cif_files += 1
            fixed_families += result.fixed_families
            fixed_hole_regions += result.fixed_hole_regions
            saved_count += 1
        else:
            unchanged_cif_files += 1
        del result
    return FixInternalContoursCifJobSummary(
        run_id=run_id,
        stats=InternalContourFixStats(
            checked_cif_files=checked_cif_files,
            checked_families=checked_families,
            fixed_cif_files=fixed_cif_files,
            fixed_families=fixed_families,
            fixed_hole_regions=fixed_hole_regions,
            skipped_klayout_keyholes=skipped_klayout_keyholes,
            unchanged_cif_files=unchanged_cif_files,
            failed=tuple(failed),
            cancelled=cancelled,
        ),
        saved_count=saved_count,
    )


class FixInternalContoursCifRunnable(QRunnable):
    def __init__(
        self,
        *,
        items: tuple[FixInternalContoursCifWorkItem, ...],
        signals: FixInternalContoursCifSignals,
        cancel_event: threading.Event,
        run_id: int,
    ) -> None:
        super().__init__()
        self._items = items
        self._signals = signals
        self._cancel_event = cancel_event
        self._run_id = int(run_id)

    def run(self) -> None:
        total = len(self._items)
        checked_cif_files = 0
        checked_families = 0
        fixed_cif_files = 0
        fixed_families = 0
        fixed_hole_regions = 0
        skipped_klayout_keyholes = 0
        unchanged_cif_files = 0
        saved_count = 0
        failed: list[str] = []
        cancelled = False
        self._signals.progress.emit(self._run_id, 0, total)
        try:
            for index, item in enumerate(self._items, start=1):
                if self._cancel_event.is_set():
                    cancelled = True
                    break
                try:
                    result = _fix_internal_contours_work_item(item, self._run_id)
                except Exception as exc:
                    failed.append(f"{Path(item.cif_path).name}: {exc}")
                    result = FixInternalContoursCifItemResult(
                        run_id=self._run_id,
                        stem=item.stem,
                        cif_path=item.cif_path,
                        image_path=item.image_path,
                        polygons=(),
                        image_size=item.image_size,
                        changed=False,
                        error=str(exc),
                    )
                checked_cif_files += 1
                checked_families += result.checked_families
                skipped_klayout_keyholes += result.skipped_klayout_keyholes
                if result.changed:
                    fixed_cif_files += 1
                    fixed_families += result.fixed_families
                    fixed_hole_regions += result.fixed_hole_regions
                    saved_count += 1
                else:
                    unchanged_cif_files += 1
                if result.changed or result.error:
                    self._signals.item_finished.emit(result)
                self._signals.progress.emit(self._run_id, index, total)
            if self._cancel_event.is_set():
                cancelled = True
        finally:
            self._signals.finished.emit(
                FixInternalContoursCifJobSummary(
                    run_id=self._run_id,
                    stats=InternalContourFixStats(
                        checked_cif_files=checked_cif_files,
                        checked_families=checked_families,
                        fixed_cif_files=fixed_cif_files,
                        fixed_families=fixed_families,
                        fixed_hole_regions=fixed_hole_regions,
                        skipped_klayout_keyholes=skipped_klayout_keyholes,
                        unchanged_cif_files=unchanged_cif_files,
                        failed=tuple(failed),
                        cancelled=cancelled,
                    ),
                    saved_count=saved_count,
                )
            )


def _fix_internal_contours_work_item(
    item: FixInternalContoursCifWorkItem,
    run_id: int,
) -> FixInternalContoursCifItemResult:
    image_path = item.image_path
    image_size = item.image_size
    if item.polygons is not None and image_size is not None:
        polygons = [polygon.clone() for polygon in item.polygons]
    else:
        loaded_name, loaded_size, loaded_polygons = load_polygons_vector(item.cif_path)
        polygons = loaded_polygons
        if image_size is None:
            image_size = loaded_size
        if image_path is None and loaded_name:
            image_path = loaded_name
    if image_size is None:
        raise ValueError(tr("cif_size_header_missing", path=item.cif_path))
    if image_path is None:
        image_path = str(Path(item.cif_path).with_suffix(""))

    fixed_polygons, analysis, changed = fix_internal_contour_display(polygons, image_size)
    if changed:
        save_polygons_vector(
            item.cif_path,
            image_path,
            fixed_polygons,
            image_size=image_size,
            cutout_display=True,
        )
    return FixInternalContoursCifItemResult(
        run_id=run_id,
        stem=item.stem,
        cif_path=item.cif_path,
        image_path=image_path,
        polygons=tuple(polygon.clone() for polygon in fixed_polygons),
        image_size=image_size,
        changed=changed,
        fixed_families=len(analysis.issues) if changed else 0,
        fixed_hole_regions=sum(issue.hole_centers_filled for issue in analysis.issues) if changed else 0,
        checked_families=analysis.checked_families,
        skipped_klayout_keyholes=analysis.skipped_klayout_keyholes,
    )
