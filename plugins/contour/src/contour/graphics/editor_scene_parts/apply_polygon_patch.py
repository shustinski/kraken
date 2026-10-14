"""editor_scene: apply polygon patch."""
from __future__ import annotations
from ...application.vector_geometry_postprocess import (
    overlap_check_roots_for_layer_patch,
    patch_polygons_needing_repair,
)
from contour.domain import PolygonData


class ApplyPolygonPatchMixin:
    def _apply_polygon_patch(
        self,
        *,
        remove_ids: list[int],
        add_polygons: list[PolygonData],
        emit_signal: bool = True,
        select_polygon_id: int | None = None,
        apply_selection: bool = False,
        sync_overlap_repair: bool = False,
    ) -> None:
        """Remove and add polygons, refreshing only affected scene items."""

        if not self._applying_deferred_geometry_repair:
            self._cancel_deferred_geometry_repair()
        self.end_zoom_vector_render_mode()
        prev_primary = self._selected_polygon_id
        prev_selected_ids = set(self._selected_polygon_ids)
        removed_id_set = set(remove_ids)
        refresh_ids: set[int] = set()
        removed_snapshots = [
            self._polygons[polygon_id].clone()
            for polygon_id in remove_ids
            if polygon_id in self._polygons
        ]
        for polygon_id in remove_ids:
            refresh_ids.update(self._polygon_edit_family_ids(polygon_id))

        for polygon_id in remove_ids:
            self._remove_polygon_internal(polygon_id, emit_signal=False, refresh=False)

        for polygon in add_polygons:
            self._add_polygon_internal(
                polygon,
                emit_signal=False,
                refresh=False,
                paint=False,
                defer_pick_z_rebuild=True,
            )
            refresh_ids.update(self._polygon_edit_family_ids(polygon.id))

        self._discard_authored_cif_paint_rings(
            {polygon.id for polygon in add_polygons} | removed_id_set
        )

        for polygon_id in list(refresh_ids):
            root_id = self._polygon_root_id(polygon_id)
            if root_id is not None:
                refresh_ids.update(self._polygon_family_ids(root_id))

        current_polygons = list(self._polygons.values())
        overlap_check_roots = overlap_check_roots_for_layer_patch(
            current_polygons,
            removed_snapshots,
            add_polygons,
        )
        self._polygons_needing_repair = patch_polygons_needing_repair(
            self._polygons_needing_repair,
            current_polygons,
            removed_ids=removed_id_set,
            removed_polygons=removed_snapshots,
            added_or_changed=add_polygons,
            settings=self._vector_geometry_settings,
            include_overlap=sync_overlap_repair,
        )
        if sync_overlap_repair:
            self.apply_polygons_needing_repair(self._polygons_needing_repair, refresh_items=True)
        else:
            self._schedule_overlap_repair_patch(overlap_check_roots)
        self._rebuild_outer_pick_z_ranks()
        refresh_ids.update(self._rebuild_object_colors())

        if apply_selection and select_polygon_id is not None and select_polygon_id in self._polygons:
            self._selected_polygon_ids = {select_polygon_id}
            self._selected_polygon_id = select_polygon_id
        else:
            new_ids = set(self._polygons)
            preserved_sorted = sorted(prev_selected_ids & new_ids)
            if preserved_sorted:
                self._selected_polygon_ids = set(preserved_sorted)
                self._selected_polygon_id = (
                    prev_primary if prev_primary in preserved_sorted else preserved_sorted[0]
                )
            elif prev_primary in removed_id_set or prev_selected_ids & removed_id_set:
                self._selected_polygon_id = None
                self._selected_polygon_ids.clear()

        editable_vertex_ids = self._editable_vertex_polygon_ids()
        for polygon_id in refresh_ids:
            if polygon_id in self._polygon_items:
                self._refresh_polygon_item(
                    polygon_id,
                    editable_vertex_ids=editable_vertex_ids,
                )

        if emit_signal:
            self.polygonsChanged.emit()
            self.activePolygonChanged.emit(self._selected_polygon_id)
