"""editor_scene: bulk restore polygons."""
from __future__ import annotations
from contour.domain import PolygonData
from contour.graphics.brush_vector import (
    polygon_equivalent_preserved,
)


class BulkRestorePolygonsMixin:
    def _bulk_restore_polygons(
        self,
        polygons: list[PolygonData],
        *,
        emit_signal: bool = True,
        selection_fallback: PolygonData | None = None,
    ) -> None:
        self._cancel_deferred_geometry_repair()
        self.end_zoom_vector_render_mode()
        prev_primary = self._selected_polygon_id
        prev_selected_ids = set(self._selected_polygon_ids)
        self._recycle_polygon_items()
        self._polygons.clear()
        self._polygon_spatial_index.clear()
        self._polygon_hover_contours.clear()
        self._hole_children_by_parent.clear()
        self._polygon_child_ids_by_parent.clear()
        self._object_colors.clear()
        self._hover_conductor_polygon_id = None
        self._vertex_preview_polygon_id = None
        self._selected_polygon_id = None
        self._selected_polygon_ids.clear()
        self._next_polygon_id = 1
        for polygon in polygons:
            self._add_polygon_internal(
                polygon,
                emit_signal=False,
                refresh=False,
                paint=False,
                defer_pick_z_rebuild=True,
            )
        self._start_recycled_polygon_cleanup()
        if polygons:
            self._next_polygon_id = max(polygon.id for polygon in polygons) + 1
            new_ids = {polygon.id for polygon in polygons}
            preserved_sorted = sorted(prev_selected_ids & new_ids)
            if preserved_sorted:
                self._selected_polygon_ids = set(preserved_sorted)
                self._selected_polygon_id = (
                    prev_primary if prev_primary in preserved_sorted else preserved_sorted[0]
                )
            elif selection_fallback is not None:
                for polygon in polygons:
                    if polygon_equivalent_preserved(polygon, [selection_fallback]):
                        self._selected_polygon_id = polygon.id
                        self._selected_polygon_ids = {polygon.id}
                        break
                else:
                    self._selected_polygon_id = None
                    self._selected_polygon_ids.clear()
            else:
                self._selected_polygon_id = None
                self._selected_polygon_ids.clear()
        self._refresh_all_items(defer_overlap_repair=True)
        if emit_signal:
            self.polygonsChanged.emit()
            self.activePolygonChanged.emit(self._selected_polygon_id)
