"""processing_mixin: begin fix internal contours."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    FixInternalContoursCifWorkItem,
    InternalContourFixStats,
    QTimer,
    threading,
)


class BeginFixInternalContoursMixin:
    def _begin_fix_internal_contours_batch(
        self: Any,
        work_items: tuple[FixInternalContoursCifWorkItem, ...],
        *,
        run_id: int,
        cancel_event: threading.Event,
    ) -> None:
        self._fix_internal_contours_pending_items = list(work_items)
        self._fix_internal_contours_batch_run_id = int(run_id)
        self._fix_internal_contours_batch_cancel = cancel_event
        self._fix_internal_contours_batch_total = len(work_items)
        self._fix_internal_contours_batch_index = 0
        self._fix_internal_contours_batch_failed: list[str] = []
        self._fix_internal_contours_batch_stats = InternalContourFixStats()
        self._fix_internal_contours_batch_saved_count = 0
        QTimer.singleShot(0, self._step_fix_internal_contours_batch)
