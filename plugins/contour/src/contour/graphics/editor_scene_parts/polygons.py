"""editor_scene: polygons."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QGraphicsScene,
)
from contour.domain import PolygonData
from contour.graphics.tool_mode_logic import (
    is_via_polygon as _is_via_polygon,
)


class PolygonsMixin:
    def set_polygons(
        self,
        polygons: list[PolygonData],
        *,
        emit_signal: bool = True,
        repair_reasons: dict[int, list[str]] | None = None,
        scan_repair: bool = True,
        defer_hit_paths: bool = False,
        defer_object_colors: bool = False,
    ) -> None:
        self.end_zoom_vector_render_mode()
        self.undo_stack.clear()
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
        if repair_reasons is not None:
            self._polygons_needing_repair = {
                int(polygon_id): list(codes) for polygon_id, codes in repair_reasons.items()
            }
        elif not scan_repair:
            self._polygons_needing_repair = {}
        will_refresh = bool(polygons) and (
            any(polygon.is_hole for polygon in polygons)
            or not all(_is_via_polygon(polygon) for polygon in polygons)
        )
        previous_index = self.itemIndexMethod()
        self.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        try:
            for polygon in polygons:
                self._add_polygon_internal(
                    polygon,
                    emit_signal=False,
                    refresh=False,
                    paint=not will_refresh,
                    defer_pick_z_rebuild=True,
                )
            self._start_recycled_polygon_cleanup()
            if polygons:
                self._next_polygon_id = max(polygon.id for polygon in polygons) + 1
            if will_refresh:
                self._refresh_all_items(
                    rebuild_repair_cache=scan_repair and repair_reasons is None,
                    defer_hit_paths=defer_hit_paths,
                    defer_object_colors=defer_object_colors,
                )
            elif polygons:
                if scan_repair and repair_reasons is None:
                    self._rebuild_polygons_needing_repair_cache()
                self._rebuild_outer_pick_z_ranks()
                for polygon_id, item in self._polygon_items.items():
                    item.setZValue(self._pick_z_value_for_polygon(self._polygons[polygon_id]))
        finally:
            self.setItemIndexMethod(previous_index)
        if emit_signal:
            self.polygonsChanged.emit()
            self.activePolygonChanged.emit(self._selected_polygon_id)
