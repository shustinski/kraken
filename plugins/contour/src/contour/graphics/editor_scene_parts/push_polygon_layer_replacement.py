"""editor_scene: push polygon layer replacement."""
from __future__ import annotations
from dataclasses import replace
from time import perf_counter
from contour.domain import PolygonData


class PushPolygonLayerReplacementMixin:
    def _push_polygon_layer_replacement(
        self,
        *,
        remove_polygon_ids: list[int],
        add_polygons: list[PolygonData],
        description: str,
        run_postprocess: bool = True,
        select_polygon_id: int | None = None,
    ) -> bool:
        """Apply one polygon-layer edit with a single scene refresh for undo/redo."""

        remove_ids = set(remove_polygon_ids)
        removed_snapshots = [
            self._polygons[polygon_id].clone()
            for polygon_id in remove_polygon_ids
            if polygon_id in self._polygons
        ]
        work: list[PolygonData] = [
            self._polygons[polygon_id]
            for polygon_id in sorted(self._polygons)
            if polygon_id not in remove_ids
        ]
        changed_polygon_ids = {polygon.id for polygon in add_polygons}
        work.extend(polygon.clone() for polygon in add_polygons)
        if run_postprocess:
            profile = self._polygon_change_profile
            phase_started_at = perf_counter()
            # The outer noise-size threshold is an import/recognition control.
            # Applying it to a deliberate editor gesture can immediately
            # delete a valid user-created polygon (the default is 60,000 px²).
            edit_settings = replace(
                self._vector_geometry_settings,
                min_outer_area_px2=0.0,
            )
            remaining, _changed, accepted = self._apply_vector_postprocess_polygons(
                work,
                changed_polygon_ids=changed_polygon_ids,
                settings=edit_settings,
            )
            self._note_polygon_change_phase(profile, "postprocess", phase_started_at)
            if not accepted:
                self.warn_invalid_polygon_geometry()
                return False
        else:
            remaining = work
        unchanged_ids = {polygon_id for polygon_id in self._polygons if polygon_id not in remove_ids}
        added_for_patch = [polygon for polygon in remaining if polygon.id not in unchanged_ids]
        if not added_for_patch and not removed_snapshots:
            self.warn_invalid_polygon_geometry()
            return False
        self._push_known_polygon_patch(
            removed_polygons=removed_snapshots,
            added_polygons=added_for_patch,
            description=description,
            select_polygon_id=select_polygon_id,
        )
        return True
