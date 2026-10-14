"""processing_mixin: complete fix internal contours."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    FixInternalContoursCifJobSummary,
    InternalContourFixStats,
)


class CompleteFixInternalContoursMixin:
    def _complete_fix_internal_contours_batch(self: Any, *, cancelled: bool) -> None:
        summary = FixInternalContoursCifJobSummary(
            run_id=int(getattr(self, "_fix_internal_contours_batch_run_id", 0)),
            stats=InternalContourFixStats(
                checked_cif_files=self._fix_internal_contours_batch_stats.checked_cif_files,
                checked_families=self._fix_internal_contours_batch_stats.checked_families,
                fixed_cif_files=self._fix_internal_contours_batch_stats.fixed_cif_files,
                fixed_families=self._fix_internal_contours_batch_stats.fixed_families,
                fixed_hole_regions=self._fix_internal_contours_batch_stats.fixed_hole_regions,
                skipped_klayout_keyholes=self._fix_internal_contours_batch_stats.skipped_klayout_keyholes,
                unchanged_cif_files=self._fix_internal_contours_batch_stats.unchanged_cif_files,
                failed=tuple(getattr(self, "_fix_internal_contours_batch_failed", [])),
                cancelled=cancelled,
            ),
            saved_count=int(getattr(self, "_fix_internal_contours_batch_saved_count", 0)),
        )
        self._fix_internal_contours_pending_items = []
        self._on_fix_internal_contours_finished(summary)
