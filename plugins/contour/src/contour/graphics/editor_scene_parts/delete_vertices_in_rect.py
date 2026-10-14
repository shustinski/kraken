"""editor_scene: delete vertices in rect."""
from __future__ import annotations
from time import perf_counter
from PyQt6.QtCore import QPointF, QRectF
from ...application.vector_geometry_postprocess import (
    apply_delete_area_geometry_repair,
)
from contour.domain import PolygonData, compute_polygon_metrics
from ...domain.polygon_ring import collapse_redundant_polyline_vertices


class DeleteVerticesInRectMixin:
    def delete_vertices_in_rect(self, rect: QRectF) -> int:
        if not self.can_add_polygon():
            return 0
        normalized = rect.normalized()
        if normalized.width() < 1.0 and normalized.height() < 1.0:
            return 0
        min_ring_vertices = 3
        remaining_by_id: dict[int, list[tuple[float, float]]] = {}
        for polygon_id in sorted(self._polygons):
            polygon = self._polygons[polygon_id]
            remaining_points = [
                (float(x_coord), float(y_coord))
                for x_coord, y_coord in polygon.points
                if not normalized.contains(QPointF(x_coord, y_coord))
            ]
            if len(remaining_points) == len(polygon.points):
                continue
            remaining_by_id[polygon_id] = collapse_redundant_polyline_vertices(
                remaining_points,
                closed=True,
                min_vertices=0,
            )

        ids_to_delete: set[int] = set()
        for polygon_id, remaining_points in remaining_by_id.items():
            if len(remaining_points) >= min_ring_vertices:
                continue
            polygon = self._polygons[polygon_id]
            if polygon.is_hole:
                ids_to_delete.add(polygon_id)
            else:
                ids_to_delete.update(self._polygon_family_ids(polygon_id))

        point_updates = [
            (polygon_id, remaining_points)
            for polygon_id, remaining_points in remaining_by_id.items()
            if polygon_id not in ids_to_delete and len(remaining_points) >= min_ring_vertices
        ]
        if not point_updates and not ids_to_delete:
            return 0

        with self._polygon_change_profile_scope("delete_vertices") as profile:
            phase_started_at = perf_counter()
            before = self.get_polygons()
            remaining: list[PolygonData] = []
            deleted = 0
            remaining_lookup = dict(point_updates)
            for polygon in before:
                if polygon.id in ids_to_delete:
                    deleted += len(polygon.points)
                    continue
                clone = polygon.clone()
                if clone.id in remaining_lookup:
                    updated_points = remaining_lookup[clone.id]
                    deleted += len(clone.points) - len(updated_points)
                    clone.points = updated_points
                    area, perimeter, bbox = compute_polygon_metrics(clone.points)
                    clone.area = float(area)
                    clone.perimeter = float(perimeter)
                    clone.bbox = bbox
                remaining.append(clone)
            self._note_polygon_change_phase(profile, "collect", phase_started_at)
            phase_started_at = perf_counter()
            changed_polygon_ids = set(remaining_lookup)
            removed_snapshots = [
                polygon.clone() for polygon in before if polygon.id in ids_to_delete
            ]
            remaining = apply_delete_area_geometry_repair(
                remaining,
                self._vector_geometry_settings,
                changed_polygon_ids=changed_polygon_ids,
                removed_ids=ids_to_delete,
                removed_polygons=removed_snapshots,
            )
            self._note_polygon_change_phase(profile, "repair", phase_started_at)
            phase_started_at = perf_counter()
            self._push_polygon_set_change(
                before,
                remaining,
                "Delete vertices in area",
                sync_overlap_repair=True,
            )
            self._note_polygon_change_phase(profile, "undo_commit", phase_started_at)
            if profile is not None:
                profile.set_metadata(deleted_vertices=deleted, polygons_removed=len(ids_to_delete))
            return deleted
