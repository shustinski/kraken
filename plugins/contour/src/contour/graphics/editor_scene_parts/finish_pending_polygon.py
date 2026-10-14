"""editor_scene: finish pending polygon."""
from __future__ import annotations
from math import hypot
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import (
    QPainterPath,
)
from contour.domain import PolygonData, compute_polygon_metrics, integer_point, integer_points
from ...domain.polygon_ring import collapse_redundant_polyline_vertices
from contour.graphics.geometry import (
    _measurement_label_position,
    is_valid_open_polyline_last_edge,
)
from contour.graphics.polygon_creation import (
    POLYGON_COMMIT_INVALID_RING,
    POLYGON_COMMIT_TOO_FEW_VERTICES,
    polygon_commit_acceptability,
)


class FinishPendingPolygonMixin:
    def preview_edge_move(
        self,
        polygon_id: int,
        edge_index: int,
        origin_points: list[tuple[float, float]],
        delta: tuple[float, float],
    ) -> list[tuple[float, float]]:
        moved = self._translate_edge_points(origin_points, edge_index, delta)
        if self._polygon_edit_preview_id != polygon_id:
            self.begin_polygon_edit_preview(polygon_id)
        self._polygon_edit_preview_points = moved
        self._update_polygon_edit_preview_path()
        return list(moved)

    def preview_polygon_move(
        self,
        polygon_id: int,
        points: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if self._polygon_edit_preview_id != polygon_id:
            self.begin_polygon_edit_preview(polygon_id)
        self._polygon_edit_preview_points = integer_points(points)
        origin = self._polygon_edit_preview_origin_points
        if origin and self._polygon_edit_preview_points:
            dx = self._polygon_edit_preview_points[0][0] - origin[0][0]
            dy = self._polygon_edit_preview_points[0][1] - origin[0][1]
            self._polygon_edit_preview_item.setPos(float(dx), float(dy))
        else:
            self._update_polygon_edit_preview_path()
        return list(self._polygon_edit_preview_points)

    def start_pending_polygon(self, *, for_brush: bool = False) -> None:
        self._pending_points.clear()
        self._pending_cursor = None
        self._pending_polyline_for_brush = bool(for_brush)
        self._update_pending_path()

    def append_brush_vertex(self, scene_pos: QPointF, brush_diameter: float) -> None:
        del brush_diameter
        nx, ny = integer_point((scene_pos.x(), scene_pos.y()))
        # Screen-pixel spacing is enforced in the view; keep only exact duplicates here.
        if self._pending_points and hypot(nx - self._pending_points[-1][0], ny - self._pending_points[-1][1]) < 1e-6:
            return
        self._pending_points.append((nx, ny))
        self._update_pending_path()

    def replace_pending_points(self, points: list[tuple[float, float]]) -> None:
        self._pending_points = integer_points(points)
        self._pending_cursor = None
        self._update_pending_path()

    def append_pending_point(self, scene_pos: QPointF) -> None:
        point = integer_point((scene_pos.x(), scene_pos.y()))
        if (
            self._pending_points
            and hypot(point[0] - self._pending_points[-1][0], point[1] - self._pending_points[-1][1]) < 1.0
        ):
            return
        trial = [*self._pending_points, point]
        if not self._pending_polyline_for_brush and not is_valid_open_polyline_last_edge(trial):
            self.warn_invalid_polygon_geometry()
            return
        self._pending_points.append(point)
        self._pending_points = collapse_redundant_polyline_vertices(
            self._pending_points,
            closed=False,
            min_vertices=2,
        )
        self._update_pending_path()

    def update_pending_cursor(self, scene_pos: QPointF) -> None:
        if not self._pending_points:
            return
        self._pending_cursor = integer_point((scene_pos.x(), scene_pos.y()))
        self._update_pending_path()

    def cancel_pending_polygon(self) -> None:
        self._pending_points.clear()
        self._pending_cursor = None
        self._pending_polyline_for_brush = False
        self._update_pending_path()
        self.clear_preview_rect()

    def finish_pending_polygon(self) -> bool:
        if not self.can_add_polygon():
            self.cancel_pending_polygon()
            return False
        self._pending_points = collapse_redundant_polyline_vertices(
            self._pending_points,
            closed=True,
            min_vertices=3,
        )
        acceptable, reason = polygon_commit_acceptability(self._pending_points)
        if not acceptable:
            if reason == POLYGON_COMMIT_TOO_FEW_VERTICES:
                self.cancel_pending_polygon()
                self._log_polygon_commit_rejection(reason)
                return False
            if reason == POLYGON_COMMIT_INVALID_RING:
                self.warn_invalid_polygon_geometry()
                return False
            self._log_polygon_commit_rejection(reason)
            return False
        area, perimeter, bbox = compute_polygon_metrics(self._pending_points)
        polygon = PolygonData(
            id=self._next_polygon_id,
            points=integer_points(self._pending_points),
            is_hole=False,
            parent_id=None,
            shape_hint="manual_outline",
            area=area,
            perimeter=perimeter,
            bbox=bbox,
        )
        clipped_polygons = self._clip_authored_polygon_to_image(polygon)
        if not clipped_polygons:
            self.cancel_pending_polygon()
            return False
        clipped_polygons = [
            clipped_polygon
            for clipped_polygon in clipped_polygons
            if not self._is_small_inner_candidate(clipped_polygon.points, thickness=None)
        ]
        if not clipped_polygons:
            self.cancel_pending_polygon()
            return False
        for clipped_polygon in clipped_polygons:
            self._add_or_merge_polygon(clipped_polygon)
        self.cancel_pending_polygon()
        return True

    def pending_last_point(self) -> QPointF | None:
        if not self._pending_points:
            return None
        x_coord, y_coord = self._pending_points[-1]
        return QPointF(x_coord, y_coord)

    def pending_points_snapshot(self) -> list[tuple[float, float]]:
        return [(float(x_coord), float(y_coord)) for x_coord, y_coord in self._pending_points]

    def has_pending_polygon(self) -> bool:
        return bool(self._pending_points)

    def set_preview_rect(self, start: QPointF, end: QPointF) -> None:
        rect = QRectF(start, end).normalized()
        path = QPainterPath()
        path.addRect(rect)
        self._preview_rect_item.setPath(path)

    def preview_delete_vertices_in_rect(self, start: QPointF, end: QPointF) -> None:
        rect = QRectF(start, end).normalized()
        self.set_preview_rect(start, end)
        if rect.width() < 1.0 and rect.height() < 1.0:
            self._set_delete_area_highlight_ids(set())
            return
        highlighted = {
            polygon_id
            for polygon_id, polygon in self._polygons.items()
            if any(rect.contains(QPointF(x_coord, y_coord)) for x_coord, y_coord in polygon.points)
        }
        self._set_delete_area_highlight_ids(highlighted)

    def _set_delete_area_highlight_ids(self, highlighted: set[int]) -> None:
        changed_ids = self._delete_area_highlight_ids ^ highlighted
        if not changed_ids:
            return
        self._delete_area_highlight_ids = set(highlighted)
        self._refresh_polygon_items_by_id(*changed_ids)

    def clear_preview_rect(self) -> None:
        self._preview_rect_item.setPath(QPainterPath())
        self._set_delete_area_highlight_ids(set())

    def set_measurement(self, start: QPointF, end: QPointF, label_text: str = "") -> None:
        path = QPainterPath()
        path.moveTo(start)
        path.lineTo(end)
        self._measurement_item.setPath(path)
        self._set_measurement_marker(self._measurement_start_marker, start)
        self._set_measurement_marker(self._measurement_end_marker, end)
        if label_text:
            self._measurement_label_item.setText(label_text)
            self._measurement_label_item.setPos(_measurement_label_position(start, end))
            self._measurement_label_item.show()
        else:
            self._measurement_label_item.hide()

    def clear_measurement(self) -> None:
        self._measurement_item.setPath(QPainterPath())
        self._measurement_start_marker.hide()
        self._measurement_end_marker.hide()
        self._measurement_label_item.hide()
