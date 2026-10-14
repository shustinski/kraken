"""editor_scene: touched polygon ids."""
from __future__ import annotations
from contour.domain import PolygonData, compute_polygon_metrics, integer_points
from contour.graphics.brush_vector import (
    QUAD_SEGS_BRUSH_DEFAULT,
    PreservedPolygonMatchCache,
    polygon_equivalent_preserved,
    polygon_footprint_geom,
    tool_geometry,
)
from contour.graphics.geometry import (
    _bboxes_intersect,
    _smallest_containing_polygon,
)


class TouchedPolygonIdsMixin:
    def _touched_polygon_ids(
        self,
        candidate_ids: list[int],
        shape_bbox: tuple[int, int, int, int],
        points: list[tuple[float, float]],
        thickness: float | None,
    ) -> set[int]:
        touched_ids: set[int] = set()
        try:
            tool_shape = tool_geometry(points, thickness, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
            buffered_tool = tool_shape.buffer(1e-7)
        except Exception:
            return touched_ids

        if tool_shape.is_empty:
            return touched_ids

        for polygon_id in candidate_ids:

            polygon = self._polygons.get(polygon_id)

            if polygon is None or not _bboxes_intersect(shape_bbox, polygon.bbox):
                continue
            polygon_region = polygon_footprint_geom(polygon.points)

            if polygon_region.is_empty:

                continue
            if polygon_region.intersects(buffered_tool):
                touched_ids.add(polygon_id)
        return touched_ids

    def _preserved_polygons(
        self,
        render_ids: list[int],
        touched_ids: set[int],
        root_ids: list[int],
    ) -> list[PolygonData]:
        root_id_set = set(root_ids)
        return [
            self._polygons[polygon_id].clone()
            for polygon_id in render_ids
            if polygon_id not in root_id_set and polygon_id not in touched_ids
        ]

    def _polygon_family_ids(self, polygon_id: int) -> list[int]:
        family_ids: list[int] = []
        pending = [polygon_id]
        while pending:
            current_id = pending.pop()
            if current_id in family_ids or current_id not in self._polygons:
                continue
            family_ids.append(current_id)
            pending.extend(self._polygon_child_ids_by_parent.get(current_id, ()))
        return sorted(family_ids)

    def _assign_polygon_ids(
        self,
        polygons: list[PolygonData],
        replaced_ids: list[int],
        *,
        reserved_ids: set[int] | None = None,
    ) -> list[PolygonData]:
        reserved = reserved_ids or set()
        reusable_ids = [
            polygon_id
            for polygon_id in sorted(
                replaced_ids,
                key=lambda current_id: (
                    -abs(float(self._polygons[current_id].area)) if current_id in self._polygons else 0.0,
                    current_id,
                ),
            )
            if polygon_id not in reserved
        ]
        sorted_polygons = sorted(polygons, key=lambda polygon: -abs(float(polygon.area)))
        allocated_ids = self._allocate_polygon_ids(reusable_ids, len(sorted_polygons))
        id_map = {
            polygon.id: allocated_id for polygon, allocated_id in zip(sorted_polygons, allocated_ids, strict=False)
        }
        for polygon, allocated_id in zip(sorted_polygons, allocated_ids, strict=False):
            polygon.parent_id = None if polygon.parent_id is None else id_map.get(polygon.parent_id)
            polygon.id = allocated_id
        return sorted(sorted_polygons, key=lambda polygon: polygon.id)

    def _restore_preserved_polygons(
        self,
        rebuilt_polygons: list[PolygonData],
        deleted_ids: list[int],
        preserved_polygons: list[PolygonData],
    ) -> list[PolygonData]:
        if not preserved_polygons:
            return self._assign_polygon_ids(rebuilt_polygons, deleted_ids)
        match_cache = PreservedPolygonMatchCache.from_polygons(preserved_polygons)
        filtered_rebuilt = [
            polygon
            for polygon in rebuilt_polygons
            if not polygon_equivalent_preserved(polygon, preserved_polygons, match_cache=match_cache)
        ]
        assigned_rebuilt = self._assign_polygon_ids(
            filtered_rebuilt,
            deleted_ids,
            reserved_ids={polygon.id for polygon in preserved_polygons},
        )
        restored_polygons = assigned_rebuilt + [polygon.clone() for polygon in preserved_polygons]
        self._repair_preserved_parent_links(restored_polygons, preserved_polygons)
        return sorted(restored_polygons, key=lambda polygon: polygon.id)

    def _matches_any_preserved_polygon(self, polygon: PolygonData, preserved_polygons: list[PolygonData]) -> bool:
        return polygon_equivalent_preserved(polygon, preserved_polygons)

    def _repair_preserved_parent_links(
        self,
        polygons: list[PolygonData],
        preserved_polygons: list[PolygonData],
    ) -> None:
        non_hole_polygons = [polygon for polygon in polygons if not polygon.is_hole]
        for preserved in preserved_polygons:
            restored = next((polygon for polygon in polygons if polygon.id == preserved.id), None)
            if restored is None:
                continue
            parent = _smallest_containing_polygon(restored, non_hole_polygons)
            restored.parent_id = None if parent is None else parent.id

    def _create_polygon_snapshot(self, polygon_id: int, points: list[tuple[float, float]]) -> PolygonData:
        existing = self._polygons[polygon_id]
        area, perimeter, bbox = compute_polygon_metrics(points)
        return PolygonData(
            id=existing.id,
            points=integer_points(points),
            is_hole=existing.is_hole,
            parent_id=existing.parent_id,
            category=existing.category,
            shape_hint=existing.shape_hint,
            area=area,
            perimeter=perimeter,
            bbox=bbox,
            recognition_score=existing.recognition_score,
            # A recognition rejection describes the pre-edit geometry.  If it
            # is retained, a manually repaired ring remains painted as invalid
            # forever even though its authored geometry has changed.
            reject_reason="",
            # The authored CIF keyhole ring describes the geometry before this
            # edit.  Keeping it would make display ignore the edited family.
            cif_paint_ring=[],
        )
