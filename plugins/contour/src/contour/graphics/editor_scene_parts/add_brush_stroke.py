"""editor_scene: add brush stroke."""
from __future__ import annotations
from time import perf_counter


class AddBrushStrokeMixin:
    def add_brush_stroke(self, points: list[tuple[float, float]], thickness: float, erase: bool = False) -> bool:
        if not self.can_add_polygon():
            self.cancel_pending_polygon()
            return False
        if len(points) < 1:
            self.cancel_pending_polygon()
            return False
        if erase:
            if self._is_small_inner_candidate(list(points), thickness=thickness):
                self.cancel_pending_polygon()
                return False
            changed = self._subtract_shape_from_scene(points=list(points), thickness=thickness, label="Erase brush stroke")
            self.cancel_pending_polygon()
            return changed
        if self._is_small_inner_candidate(list(points), thickness=thickness):
            self.cancel_pending_polygon()
            return False
        with self._polygon_change_profile_scope("add_brush") as profile:
            phase_started_at = perf_counter()
            merged_polygons, overlapping_ids = self._merge_shape_into_scene(points=list(points), thickness=thickness)
            self._note_polygon_change_phase(profile, "merge", phase_started_at)
            if merged_polygons is None:
                self.cancel_pending_polygon()
                return False
            if not merged_polygons:
                self.cancel_pending_polygon()
                return False
            phase_started_at = perf_counter()
            merged_polygons = self._clip_authored_polygons_to_image(merged_polygons)
            self._note_polygon_change_phase(profile, "clip", phase_started_at)
            if not merged_polygons:
                self.cancel_pending_polygon()
                return False
            if profile is not None:
                profile.set_metadata(overlapping=len(overlapping_ids), result_polygons=len(merged_polygons))
            phase_started_at = perf_counter()
            if not self._push_polygon_layer_replacement(
                remove_polygon_ids=overlapping_ids,
                add_polygons=merged_polygons,
                description="Add brush stroke",
                select_polygon_id=merged_polygons[0].id,
            ):
                self.cancel_pending_polygon()
                return False
            self._note_polygon_change_phase(profile, "undo_commit", phase_started_at)
        self.cancel_pending_polygon()
        return True
