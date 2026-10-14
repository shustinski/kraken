"""editor_scene: subtract pending polygon."""
from __future__ import annotations
from contour.graphics.polygon_creation import (
    POLYGON_COMMIT_INVALID_RING,
    POLYGON_COMMIT_TOO_FEW_VERTICES,
    polygon_commit_acceptability,
)


class SubtractPendingPolygonMixin:
    def subtract_pending_polygon(self) -> bool:
        if not self.can_add_polygon():
            self.cancel_pending_polygon()
            return False
        acceptable, reason = polygon_commit_acceptability(self._pending_points)
        if not acceptable:
            if reason == POLYGON_COMMIT_TOO_FEW_VERTICES:
                self.cancel_pending_polygon()
                return False
            if reason == POLYGON_COMMIT_INVALID_RING:
                self.warn_invalid_polygon_geometry()
                return False
            self._log_polygon_commit_rejection(reason)
            return False
        if self._is_small_inner_candidate(self._pending_points, thickness=None):
            self.cancel_pending_polygon()
            return False
        changed = self._subtract_shape_from_scene(points=self._pending_points, thickness=None, label="Erase polygon")
        self.cancel_pending_polygon()
        return changed
