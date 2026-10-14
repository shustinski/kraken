"""editor_scene: add rectangle polygon."""
from __future__ import annotations
from PyQt6.QtCore import QPointF, QRectF
from contour.domain import PolygonData, compute_polygon_metrics, integer_points
from contour.graphics.polygon_creation import (
    POLYGON_COMMIT_INVALID_RING,
    polygon_commit_acceptability,
)


class AddRectanglePolygonMixin:
    def add_rectangle_polygon(self, start: QPointF, end: QPointF, erase: bool = False) -> bool:
        if not self.can_add_polygon():
            self.clear_preview_rect()
            return False
        rect = QRectF(start, end).normalized()
        if rect.width() < 1.0 or rect.height() < 1.0:
            self.clear_preview_rect()
            return False
        points = [
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.right(), rect.bottom()),
            (rect.left(), rect.bottom()),
        ]
        points = integer_points(points)
        acceptable, reason = polygon_commit_acceptability(points)
        if not acceptable:
            self.clear_preview_rect()
            if reason == POLYGON_COMMIT_INVALID_RING:
                self.warn_invalid_polygon_geometry()
            else:
                self._log_polygon_commit_rejection(reason)
            return False
        area, perimeter, bbox = compute_polygon_metrics(points)
        polygon = PolygonData(
            id=self._next_polygon_id,
            points=points,
            is_hole=False,
            parent_id=None,
            area=area,
            perimeter=perimeter,
            bbox=bbox,
        )
        clipped_polygons = self._clip_authored_polygon_to_image(polygon, allocate_ids=not erase)
        if not clipped_polygons:
            self.clear_preview_rect()
            return False
        polygon = clipped_polygons[0]
        if erase:
            if self._is_small_inner_candidate(list(polygon.points), thickness=None):
                self.clear_preview_rect()
                return False

            erased_ok = bool(
                self._subtract_shape_from_scene(points=list(polygon.points), thickness=None, label="Erase rectangle")
            )

            self.clear_preview_rect()


            return erased_ok

        if self._is_small_inner_candidate(polygon.points, thickness=None):
            self.clear_preview_rect()
            return False

        self._add_or_merge_polygon(polygon, label="Add rectangle")

        self.clear_preview_rect()


        return True
