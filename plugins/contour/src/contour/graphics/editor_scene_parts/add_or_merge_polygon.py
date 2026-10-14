"""editor_scene: add or merge polygon."""
from __future__ import annotations
from time import perf_counter
from contour.domain import PolygonData


class AddOrMergePolygonMixin:
    def _add_or_merge_polygon(self, polygon: PolygonData, label: str = "Add polygon") -> None:
        operation = self._polygon_change_operation_from_label(label)
        with self._polygon_change_profile_scope(operation) as profile:
            if not self._polygons:
                phase_started_at = perf_counter()
                self._push_polygon_layer_replacement(
                    remove_polygon_ids=[],
                    add_polygons=[polygon],
                    description=label,
                    select_polygon_id=polygon.id,
                )
                self._note_polygon_change_phase(profile, "undo_commit", phase_started_at)
                return
            phase_started_at = perf_counter()
            merged_polygons, overlapping_ids = self._merge_shape_into_scene(points=polygon.points, thickness=None)
            self._note_polygon_change_phase(profile, "merge", phase_started_at)
            if profile is not None:
                profile.set_metadata(overlapping=len(overlapping_ids))
            if merged_polygons is None:
                phase_started_at = perf_counter()
                self._push_polygon_layer_replacement(
                    remove_polygon_ids=[],
                    add_polygons=[polygon],
                    description=label,
                    select_polygon_id=polygon.id,
                )
                self._note_polygon_change_phase(profile, "undo_commit", phase_started_at)
                return
            if not overlapping_ids:
                phase_started_at = perf_counter()
                self._push_polygon_layer_replacement(
                    remove_polygon_ids=[],
                    add_polygons=[polygon],
                    description=label,
                    select_polygon_id=polygon.id,
                )
                self._note_polygon_change_phase(profile, "undo_commit", phase_started_at)
                return
            if not merged_polygons:
                phase_started_at = perf_counter()
                self._push_polygon_layer_replacement(
                    remove_polygon_ids=[],
                    add_polygons=[polygon],
                    description=label,
                    select_polygon_id=polygon.id,
                )
                self._note_polygon_change_phase(profile, "undo_commit", phase_started_at)
                return
            phase_started_at = perf_counter()
            if not self._push_polygon_layer_replacement(
                remove_polygon_ids=overlapping_ids,
                add_polygons=merged_polygons,
                description=label,
                select_polygon_id=merged_polygons[0].id,
            ):
                return
            self._note_polygon_change_phase(profile, "undo_commit", phase_started_at)
            if profile is not None:
                profile.set_metadata(result_polygons=len(merged_polygons))
