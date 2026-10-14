"""editor_scene: add pasted contacts."""
from __future__ import annotations
from PyQt6.QtCore import QPointF, QRectF
from ...application.polygon_antialiasing import antialias_polygons
from ...application.vector_geometry_postprocess import (
    polygons_needing_repair,
    repair_invalid_polygon_descriptions,
)
from contour.commands import (
    AddPolygonsCommand,
    ReplacePolygonSetCommand,
)
from contour.domain import PolygonData, compute_polygon_metrics, integer_points
from contour.i18n import tr
from contour.graphics.geometry import (
    _polygon_data_rect,
)
from contour.graphics.tool_mode_logic import (
    is_via_polygon as _is_via_polygon,
)

from contour.graphics.editor_scene_parts.contact_spatial_index import _ContactSpatialIndex


class AddPastedContactsMixin:
    def selected_polygons(self) -> list[PolygonData]:
        return [
            self._polygons[polygon_id].clone()
            for polygon_id in sorted(self._selected_polygon_ids)
            if polygon_id in self._polygons
        ]

    def selected_deletable_polygons(self) -> list[PolygonData]:
        return [polygon for polygon in self.selected_polygons() if self.polygon_is_deletable(polygon)]

    def antialias_selected_polygons(self, grade: int) -> bool:
        if not self.can_add_polygon():
            return False
        target_ids = {polygon_id for polygon_id in self._selected_polygon_ids if polygon_id in self._polygons}
        if not target_ids and self._selected_polygon_id is not None and self._selected_polygon_id in self._polygons:
            target_ids = {self._selected_polygon_id}
        if not target_ids:
            return False
        before = self.get_polygons()
        after, changed = antialias_polygons(before, grade, only_ids=target_ids)
        if not changed:
            return False
        self.undo_stack.push(ReplacePolygonSetCommand(self, before, after, "Antialias polygons"))
        self.select_polygons(sorted(target_ids))
        return True

    def repair_invalid_polygon_descriptions(self) -> bool:
        before = self.get_polygons()
        settings = self._vector_geometry_settings
        has_keyhole = any(
            (not polygon.is_hole) and polygon.description_invalid_reason() == "repeated_vertex"
            for polygon in before
        )
        if not polygons_needing_repair(before, settings) and not has_keyhole:
            return False
        after = repair_invalid_polygon_descriptions(before, settings)
        before_signature = [
            (polygon.id, polygon.is_hole, polygon.parent_id, tuple(polygon.points)) for polygon in before
        ]
        after_signature = [
            (polygon.id, polygon.is_hole, polygon.parent_id, tuple(polygon.points)) for polygon in after
        ]
        if after_signature == before_signature:
            return False
        self.undo_stack.push(
            ReplacePolygonSetCommand(
                self,
                before,
                after,
                tr("repair_invalid_polygons_undo", language=self._ui_language),
            )
        )
        return True

    def antialias_polygon(self, polygon_id: int | None, grade: int) -> bool:
        if polygon_id is None or polygon_id not in self._polygons:
            return False
        before = self.get_polygons()
        after, changed = antialias_polygons(before, grade, only_ids={polygon_id})
        if not changed:
            self.select_polygon(polygon_id)
            return False
        self.undo_stack.push(ReplacePolygonSetCommand(self, before, after, "Antialias polygon"))
        self.select_polygon(polygon_id)
        self._set_vertex_preview_polygon_id(polygon_id)
        return True

    def antialias_polygons_in_rect(self, rect: QRectF, grade: int) -> bool:
        if not self.can_add_polygon():
            return False
        normalized = rect.normalized()
        if normalized.width() <= 0.0 or normalized.height() <= 0.0:
            return False
        target_ids = {
            polygon_id
            for polygon_id, polygon in self._polygons.items()
            if _polygon_data_rect(polygon).intersects(normalized)
        }
        if not target_ids:
            return False
        before = self.get_polygons()
        after, changed = antialias_polygons(before, grade, only_ids=target_ids)
        if not changed:
            self.select_polygons(sorted(target_ids))
            return False
        self.undo_stack.push(ReplacePolygonSetCommand(self, before, after, "Antialias polygons"))
        self.select_polygons(sorted(target_ids))
        self._set_vertex_preview_polygon_id(None)
        return True

    def add_cloned_polygons_at(
        self,
        polygons: list[PolygonData],
        source_anchor: QPointF,
        target_anchor: QPointF,
    ) -> list[int]:
        if not polygons or not self.can_add_polygon_set(polygons):
            return []
        dx = target_anchor.x() - source_anchor.x()
        dy = target_anchor.y() - source_anchor.y()
        shifted_polygons: list[PolygonData] = []
        for polygon in polygons:
            shifted = polygon.clone()
            shifted.points = integer_points(
                [(float(x) + dx, float(y) + dy) for x, y in polygon.points]
            )
            shifted.area, shifted.perimeter, shifted.bbox = compute_polygon_metrics(
                shifted.points
            )
            shifted_polygons.append(shifted)
        clipped_polygons = self._clip_pasted_polygons_to_inset(shifted_polygons, inset=3.0)
        if not clipped_polygons:
            return []
        if not all(_is_via_polygon(polygon) for polygon in clipped_polygons):
            return self._merge_pasted_polygons(clipped_polygons)
        return self._add_pasted_contacts(clipped_polygons)

    def _add_pasted_contacts(self, polygons: list[PolygonData]) -> list[int]:
        new_polygons: list[PolygonData] = []
        spatial_index = _ContactSpatialIndex(self._minimum_contact_distance)
        for existing in self._polygons.values():
            if not _is_via_polygon(existing):
                continue
            existing_rect = _polygon_data_rect(existing)
            existing_center = existing_rect.center()
            spatial_index.add(
                existing_center.x(),
                existing_center.y(),
                existing_rect,
            )
        for polygon in polygons:
            candidate_rect = _polygon_data_rect(polygon)
            candidate_center = candidate_rect.center()
            center_x = candidate_center.x()
            center_y = candidate_center.y()
            if spatial_index.conflicts(
                center_x,
                center_y,
                candidate_rect,
            ):
                continue
            new_id = self._next_polygon_id
            self._next_polygon_id += 1
            new_polygon = polygon.clone()
            new_polygon.id = new_id
            new_polygon.parent_id = None
            new_polygons.append(new_polygon)
            spatial_index.add(center_x, center_y, candidate_rect)
        if not new_polygons:
            return []
        self.undo_stack.push(
            AddPolygonsCommand(
                self,
                new_polygons,
                "Paste polygons",
                select_after_redo=True,
            )
        )
        return [polygon.id for polygon in new_polygons]
