"""editor_scene: merge shape into scene."""
from __future__ import annotations
from contour.domain import PolygonData
from contour.i18n import tr
from contour.graphics.geometry import (
    _bbox_from_points,
)


class MergeShapeIntoSceneMixin:
    def _merge_shape_into_scene(
        self,
        *,
        points: list[tuple[float, float]],
        thickness: float | None,
    ) -> tuple[list[PolygonData] | None, list[int]]:
        if not points:
            return [], []
        shape_bbox = _bbox_from_points(points, padding=(round(thickness / 2.0) + 2) if thickness else 2)
        overlapping_ids = self._find_overlapping_polygon_ids(points=points, thickness=thickness, shape_bbox=shape_bbox)
        render_ids = self._render_polygon_ids(overlapping_ids)
        touched_ids = self._touched_polygon_ids(render_ids, shape_bbox, points, thickness)
        preserved_polygons = self._preserved_polygons(render_ids, touched_ids, overlapping_ids)
        merged_contours, err_msg = self._apply_tool_boolean_to_polygon_subset(
            render_ids=list(render_ids),
            points=list(points),
            thickness=thickness,
            erase=False,
        )
        if err_msg is not None:
            prefix = tr("brush_boolean_failed_log", language=self._ui_language)

            detail = err_msg.strip() or repr(err_msg)

            self.logRequested.emit(f"{prefix} ({detail})")
            return None, render_ids

        assert merged_contours is not None
        return self._restore_preserved_polygons(merged_contours, render_ids, preserved_polygons), render_ids
