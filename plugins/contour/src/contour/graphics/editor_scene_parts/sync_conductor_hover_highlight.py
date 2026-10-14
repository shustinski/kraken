"""editor_scene: sync conductor hover highlight."""
from __future__ import annotations
from time import perf_counter
from PyQt6.QtCore import QPointF
from contour.graphics.geometry import (
    resolve_conductor_hover_target_id,
    resolve_hover_polygon_id,
)


class SyncConductorHoverHighlightMixin:
    def sync_conductor_hover_highlight(self, scene_pos: QPointF) -> None:
        if not self._polygon_overlays_visible:
            self._set_hover_conductor_polygon_id(None)
            return
        candidate_ids = self._polygon_spatial_index.query_point(scene_pos.x(), scene_pos.y())
        candidate_polygons = {
            polygon_id: self._polygons[polygon_id]
            for polygon_id in candidate_ids
            if polygon_id in self._polygons
        }
        candidate_holes = {
            parent_id: [hole for hole in holes if hole.id in candidate_ids]
            for parent_id, holes in self._hole_children_by_parent.items()
            if parent_id in candidate_ids
        }
        target_id = resolve_hover_polygon_id(
            candidate_polygons,
            candidate_holes,
            scene_pos.x(),
            scene_pos.y(),
            contours_by_id=self._polygon_hover_contours,
        )
        if target_id is None:
            underneath = self.polygon_at(scene_pos)
            target_id = resolve_conductor_hover_target_id(self._polygons, underneath)
        self._set_hover_conductor_polygon_id(target_id)

    def clear_conductor_hover_highlight(self) -> None:
        self._set_hover_conductor_polygon_id(None)

    def set_show_all_editable_vertices(self, enabled: bool) -> float:
        enabled = bool(enabled)
        if enabled == self._show_all_editable_vertices:
            return 0.0
        self._show_all_editable_vertices = enabled
        if enabled:
            self._vertex_preview_polygon_id = None
        refresh_started_at = perf_counter()
        self._refresh_editable_vertices_display()
        return max(0.0, (perf_counter() - refresh_started_at) * 1000.0)

    def _refresh_editable_vertices_display(self) -> None:
        """Toggle vertex-handle visibility without rebuilding geometry or repair caches."""

        self._refresh_move_target_highlight_for_polygons(set(self._polygon_items.keys()))

    def sync_vertex_preview(self, scene_pos: QPointF) -> None:
        if self._show_all_editable_vertices or not self._polygon_overlays_visible:
            self._set_vertex_preview_polygon_id(None)
            return
        self._set_vertex_preview_polygon_id(self.polygon_at(scene_pos))

    def clear_vertex_preview(self) -> None:
        self._set_vertex_preview_polygon_id(None)

    def clear_move_target_preview(self, *, refresh: bool = True) -> None:
        self._set_move_target_preview(None, refresh=refresh)

    def sync_move_target_preview(
        self,
        scene_pos: QPointF,
        *,
        vertex_tolerance: float,
        edge_tolerance: float,
    ) -> None:
        if not self._polygon_overlays_visible:
            self._set_move_target_preview(None)
            return
        self._set_move_target_preview(
            self.pick_move_target(scene_pos, vertex_tolerance, edge_tolerance, strict=True)
        )
