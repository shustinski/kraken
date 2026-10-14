"""editor_scene: find overlapping polygon ids."""
from __future__ import annotations
from contour.domain import PolygonData
from contour.graphics.brush_vector import (
    QUAD_SEGS_BRUSH_DEFAULT,
    apply_boolean,
    bbox_intersects_geom_bounds,
    region_geometry,
    shapely_to_polygon_data_list,
    simplify_polygonal_geometry,
    tool_geometry,
)
from contour.graphics.geometry import (
    _bboxes_intersect,
)

from contour.graphics.editor_scene_parts.module_state import _LOGGER


class FindOverlappingPolygonIdsMixin:
    def _apply_tool_boolean_to_polygon_subset(
        self,
        *,
        render_ids: list[int],
        points: list[tuple[float, float]],
        thickness: float | None,
        erase: bool,
    ) -> tuple[list[PolygonData] | None, str | None]:
        try:
            brush_tool = tool_geometry(points, thickness, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
            if thickness is not None:
                brush_tool = simplify_polygonal_geometry(brush_tool)
        except Exception as exc:
            _LOGGER.exception("Failed to construct brush geometry")
            return None, f"{type(exc).__name__}: {exc}"
        try:
            base_region = region_geometry(self._polygons, render_ids)
        except ValueError:
            return None, "invalid region geometry"
        result_geom, err_msg = apply_boolean(base_region, brush_tool, subtract=erase)
        if err_msg is not None:
            return None, err_msg
        assert result_geom is not None

        polygons_list_out = shapely_to_polygon_data_list(result_geom)
        return polygons_list_out, None

    def _find_overlapping_polygon_ids(
        self,
        *,
        points: list[tuple[float, float]],
        thickness: float | None,
        shape_bbox: tuple[int, int, int, int],
    ) -> list[int]:
        overlapping_ids: list[int] = []
        try:
            tool_shape = tool_geometry(points, thickness, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
        except Exception:
            return []
        if tool_shape.is_empty:
            return []

        brush_bounds_xy = tuple(tool_shape.bounds)

        buffered_tool = tool_shape.buffer(1e-7)

        candidate_ids = self._polygon_spatial_index.query_bbox(shape_bbox)
        for polygon_id in sorted(candidate_ids):
            polygon = self._polygons.get(polygon_id)
            if polygon is None:
                continue
            if polygon.is_hole:
                continue
            if not _bboxes_intersect(shape_bbox, polygon.bbox):
                continue
            if not bbox_intersects_geom_bounds(brush_bounds_xy, polygon.bbox):
                continue
            try:
                family_geometry_shape = region_geometry(self._polygons, self._polygon_family_ids(polygon_id))
            except ValueError:
                continue
            if family_geometry_shape.intersects(buffered_tool):
                overlapping_ids.append(polygon_id)
        return overlapping_ids

    def _allocate_polygon_ids(self, overlapping_ids: list[int], count: int) -> list[int]:
        ids: list[int] = []
        reusable_ids = sorted(overlapping_ids)
        for index in range(count):
            if index < len(reusable_ids):
                ids.append(reusable_ids[index])
            else:
                ids.append(self._next_polygon_id)
                self._next_polygon_id += 1
        return ids
