"""editor_scene: apply vector postprocess polygons."""
from __future__ import annotations
from time import perf_counter
from ...application.vector_geometry_postprocess import (
    VectorGeometrySettings,
    _repair_still_invalid_ring_families,
    apply_overlap_merge_after_edit,
    apply_overlap_repair_patch,
    postprocess_after_editor_mutation,
    postprocess_after_layer_patch,
    postprocess_changed_polygon_edit,
)
from contour.commands import (
    ReplacePolygonSetCommand,
)
from contour.domain import PolygonData

from contour.graphics.editor_scene_parts.module_state import _DEFERRED_GEOMETRY_REPAIR_DELAY_MS

from contour.graphics.editor_scene_parts.gradient_arrows_path import _DeferredGeometryRepairRequest


class ApplyVectorPostprocessPolygonsMixin:
    def _schedule_overlap_repair_patch(self, overlap_check_roots: set[int]) -> None:
        if not overlap_check_roots:
            return
        self._pending_overlap_repair_roots.update(overlap_check_roots)
        self._overlap_repair_patch_timer.start()

    def _run_pending_overlap_repair_patch(self) -> None:
        overlap_check_roots = self._pending_overlap_repair_roots
        self._pending_overlap_repair_roots = set()
        if not overlap_check_roots:
            return
        previous = self._polygons_needing_repair
        updated = apply_overlap_repair_patch(
            previous,
            list(self._polygons.values()),
            overlap_check_roots,
        )
        self.apply_polygons_needing_repair(updated, refresh_items=True)

    def _cancel_deferred_geometry_repair(self) -> None:
        self._deferred_geometry_repair_timer.stop()
        self._pending_deferred_geometry_repair = None

    def pause_deferred_geometry_repair(self) -> None:
        self._deferred_geometry_repair_timer.stop()

    def resume_deferred_geometry_repair(self) -> None:
        if self._pending_deferred_geometry_repair is not None:
            self._deferred_geometry_repair_timer.start(_DEFERRED_GEOMETRY_REPAIR_DELAY_MS)

    def _schedule_deferred_geometry_repair(
        self,
        *,
        removed_ids: set[int],
        removed_polygons: list[PolygonData],
    ) -> None:
        if not removed_ids:
            return
        self._cancel_deferred_geometry_repair()
        self._pending_deferred_geometry_repair = _DeferredGeometryRepairRequest(
            removed_ids=set(removed_ids),
            removed_polygons=[polygon.clone() for polygon in removed_polygons],
        )
        self._deferred_geometry_repair_timer.start(_DEFERRED_GEOMETRY_REPAIR_DELAY_MS)

    def _run_pending_deferred_geometry_repair(self) -> None:
        request = self._pending_deferred_geometry_repair
        self._pending_deferred_geometry_repair = None
        if request is None:
            return
        before = self.get_polygons()
        repaired = apply_overlap_merge_after_edit(
            before,
            removed_ids=request.removed_ids,
            removed_polygons=request.removed_polygons,
        )
        repaired = _repair_still_invalid_ring_families(
            repaired,
            seed_polygon_ids=set(request.removed_ids),
        )
        removed, added = self._compute_polygon_layer_patch(before, repaired)
        if not removed and not added:
            return
        self._applying_deferred_geometry_repair = True
        try:
            self._apply_polygon_patch(
                remove_ids=[polygon.id for polygon in removed],
                add_polygons=added,
                emit_signal=True,
            )
        finally:
            self._applying_deferred_geometry_repair = False

    def _apply_vector_postprocess_polygons(
        self,
        polygons: list[PolygonData],
        *,
        changed_polygon_ids: set[int] | None = None,
        settings: VectorGeometrySettings | None = None,
    ) -> tuple[list[PolygonData], bool, bool]:
        active_settings = settings or self._vector_geometry_settings
        if changed_polygon_ids:
            final, changed = postprocess_after_layer_patch(
                polygons,
                active_settings,
                changed_polygon_ids=changed_polygon_ids,
                frame_width_height=None,
                include_merge=False,
            )
            return final, changed, True
        needs_full_cleanup = float(active_settings.min_hole_area_to_remove_px2) > 0.0
        if needs_full_cleanup:
            final, changed = postprocess_after_editor_mutation(
                polygons,
                active_settings,
                frame_width_height=None,
                include_merge=False,
            )
            return final, changed, True
        final, accepted, changed = postprocess_changed_polygon_edit(
            polygons,
            active_settings,
            polygon_id=self._selected_polygon_id,
        )
        if not accepted:
            return polygons, False, False
        if not changed:
            final, changed = postprocess_after_editor_mutation(
                polygons,
                active_settings,
                frame_width_height=None,
                include_merge=False,
            )
        return final, changed, True

    def _maybe_push_vector_postprocess(self, undo_text: str) -> None:
        selection_fallback = self._selection_reference_polygon()
        before = self.get_polygons()
        profile = self._polygon_change_profile
        phase_started_at = perf_counter()
        final, changed, accepted = self._apply_vector_postprocess_polygons(before)
        if not accepted:
            self.undo_stack.undo()
            self._restore_selection_fallback_if_needed(selection_fallback)
            self.warn_invalid_polygon_geometry()
            self._note_polygon_change_phase(profile, "postprocess", phase_started_at)
            return
        if changed:
            self.undo_stack.push(ReplacePolygonSetCommand(self, before, final, undo_text))
            self._restore_selection_fallback_if_needed(selection_fallback)
        self._note_polygon_change_phase(profile, "postprocess", phase_started_at)
