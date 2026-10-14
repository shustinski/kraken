"""editor_scene: subtract shape scene."""
from __future__ import annotations
from time import perf_counter
from contour.i18n import tr
from contour.graphics.geometry import (
    _bbox_from_points,
)


class SubtractShapeSceneMixin:
    def _subtract_shape_from_scene(
        self,
        *,
        points: list[tuple[float, float]],
        thickness: float | None,
        label: str,
    ) -> bool:
        if not points or not self.can_add_polygon():
            return False
        with self._polygon_change_profile_scope(self._polygon_change_operation_from_label(label)) as profile:
            phase_started_at = perf_counter()
            shape_bbox = _bbox_from_points(points, padding=(round(thickness / 2.0) + 2) if thickness else 2)
            overlapping_ids = self._find_overlapping_polygon_ids(
                points=points,
                thickness=thickness,
                shape_bbox=shape_bbox,
            )
            if not overlapping_ids:
                if profile is not None:
                    profile.set_metadata(changed=False)
                return False
            render_ids = self._render_polygon_ids(overlapping_ids)
            touched_ids = self._touched_polygon_ids(render_ids, shape_bbox, points, thickness)
            preserved_polygons = self._preserved_polygons(render_ids, touched_ids, overlapping_ids)
            self._note_polygon_change_phase(profile, "overlap_find", phase_started_at)
            if profile is not None:
                profile.set_metadata(overlapping=len(overlapping_ids))
            phase_started_at = perf_counter()
            remaining_polygons, err_msg = self._apply_tool_boolean_to_polygon_subset(
                render_ids=list(render_ids),
                points=list(points),
                thickness=thickness,
                erase=True,
            )
            self._note_polygon_change_phase(profile, "boolean", phase_started_at)
            if err_msg is not None:
                detail = err_msg.strip() or repr(err_msg)
                prefix = tr("brush_boolean_failed_log", language=self._ui_language)
                self.logRequested.emit(f"{prefix} ({detail})")
                if profile is not None:
                    profile.set_metadata(changed=False, error=detail)
                return False
            assert remaining_polygons is not None
            phase_started_at = perf_counter()
            rebuilt_polygons = self._restore_preserved_polygons(remaining_polygons, render_ids, preserved_polygons)
            self._note_polygon_change_phase(profile, "preserve_restore", phase_started_at)
            phase_started_at = perf_counter()
            select_polygon_id = rebuilt_polygons[0].id if rebuilt_polygons else None
            if not self._push_polygon_layer_replacement(
                remove_polygon_ids=render_ids,
                add_polygons=rebuilt_polygons,
                description=label,
                select_polygon_id=select_polygon_id,
            ):
                if profile is not None:
                    profile.set_metadata(changed=False)
                return False
            self._note_polygon_change_phase(profile, "undo_commit", phase_started_at)
            if profile is not None:
                profile.set_metadata(changed=True, result_polygons=len(rebuilt_polygons))
            return True
