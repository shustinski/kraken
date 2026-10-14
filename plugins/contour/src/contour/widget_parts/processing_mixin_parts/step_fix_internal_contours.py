"""processing_mixin: step fix internal contours."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    FixInternalContoursCifItemResult,
    FixInternalContoursCifWorkItem,
    InternalContourFixStats,
    Path,
    QTimer,
    _fix_internal_contours_work_item,
)

from contour.widget_parts.processing_mixin_parts.module_state import _LOGGER


class StepFixInternalContoursMixin:
    def _step_fix_internal_contours_batch(self: Any) -> None:
        if not getattr(self, "_fix_internal_contours_running", False):
            return
        if int(getattr(self, "_fix_internal_contours_batch_run_id", 0)) != int(
            getattr(self, "_fix_internal_contours_run_id", 0)
        ):
            return
        cancel_event = getattr(self, "_fix_internal_contours_batch_cancel", None)
        if cancel_event is not None and cancel_event.is_set():
            self._complete_fix_internal_contours_batch(cancelled=True)
            return

        pending: list[FixInternalContoursCifWorkItem] = getattr(self, "_fix_internal_contours_pending_items", [])
        if not pending:
            self._complete_fix_internal_contours_batch(cancelled=False)
            return

        item = pending.pop(0)
        self._fix_internal_contours_pending_items = pending
        run_id = int(self._fix_internal_contours_batch_run_id)
        try:
            result = _fix_internal_contours_work_item(item, run_id)
        except Exception as exc:
            _LOGGER.exception("Internal-contour repair failed for %s", item.cif_path)
            self._fix_internal_contours_batch_failed.append(f"{Path(item.cif_path).name}: {exc}")
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

        stats = self._fix_internal_contours_batch_stats
        stats = InternalContourFixStats(
            checked_cif_files=stats.checked_cif_files + 1,
            checked_families=stats.checked_families + result.checked_families,
            fixed_cif_files=stats.fixed_cif_files + (1 if result.changed else 0),
            fixed_families=stats.fixed_families + result.fixed_families,
            fixed_hole_regions=stats.fixed_hole_regions + result.fixed_hole_regions,
            skipped_klayout_keyholes=stats.skipped_klayout_keyholes + result.skipped_klayout_keyholes,
            unchanged_cif_files=stats.unchanged_cif_files + (0 if result.changed else 1),
            failed=tuple(self._fix_internal_contours_batch_failed),
            cancelled=False,
        )
        self._fix_internal_contours_batch_stats = stats
        if result.changed:
            self._fix_internal_contours_batch_saved_count += 1
        if result.changed or result.error:
            self._on_fix_internal_contours_item_finished(result)

        self._fix_internal_contours_batch_index = int(getattr(self, "_fix_internal_contours_batch_index", 0)) + 1
        if self._batch_progress_enabled:
            self._apply_batch_progress_bar(
                self._fix_internal_contours_batch_index,
                self._fix_internal_contours_batch_total,
                pump_events=True,
            )
        self._set_progress_status(
            "batch_progress_status",
            current=self._fix_internal_contours_batch_index,
            total=self._fix_internal_contours_batch_total,
        )
        QTimer.singleShot(0, self._step_fix_internal_contours_batch)
