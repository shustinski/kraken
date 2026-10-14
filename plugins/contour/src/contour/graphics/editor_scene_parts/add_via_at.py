"""editor_scene: add via at."""
from __future__ import annotations
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QPainterPath,
    QPen,
)
from ...application.processing import normalize_via_display_mode
from contour.commands import (
    AddPolygonCommand,
)
from contour.domain import PolygonData, compute_polygon_metrics, integer_points
from contour.graphics.geometry import (
    _centered_rect,
    _polygon_data_rect,
)
from contour.graphics.tool_mode_logic import (
    is_via_polygon as _is_via_polygon,
)


class AddViaAtMixin:
    def set_brush_cursor(self, scene_pos: QPointF | None, thickness: float, visible: bool) -> None:
        if not visible or scene_pos is None:
            self._brush_cursor_item.hide()
            return
        radius = max(1.0, float(thickness)) / 2.0
        self._brush_cursor_item.setRect(
            QRectF(scene_pos.x() - radius, scene_pos.y() - radius, radius * 2.0, radius * 2.0)
        )
        self._brush_cursor_item.show()

    def set_via_cursor(self, scene_pos: QPointF | None, width: float, height: float, visible: bool) -> None:
        if not visible or scene_pos is None:
            self._via_cursor_item.hide()
            return
        rect = _centered_rect(scene_pos, width, height)
        path = QPainterPath()
        if normalize_via_display_mode(self._display_settings.via_display_mode) == "rectangle":
            path.addRect(rect)
        else:
            path.addEllipse(rect)
        blocked = self._via_rect_overlaps_existing(rect.normalized())
        color = QColor("#EF4444" if blocked else "#A78BFA")
        pen = QPen(color, 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        brush = QColor(color)
        brush.setAlpha(42 if not blocked else 58)
        self._via_cursor_item.setPen(pen)
        self._via_cursor_item.setBrush(QBrush(brush))
        self._via_cursor_item.setPath(path)
        self._via_cursor_item.show()

    def hide_tool_cursors(self) -> None:
        self._brush_cursor_item.hide()
        self._via_cursor_item.hide()

    def add_via_at(self, scene_pos: QPointF, width: float, height: float) -> bool:
        if not self.can_add_via():
            return False
        rect = _centered_rect(scene_pos, width, height).normalized()
        if rect.width() < 1.0 or rect.height() < 1.0:
            return False
        image_rect = QRectF(self._image_rect).normalized()
        if image_rect.width() > 1.0 and image_rect.height() > 1.0 and not image_rect.contains(rect):
            return False
        if self._via_rect_overlaps_existing(rect):
            return False
        points = [
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.right(), rect.bottom()),
            (rect.left(), rect.bottom()),
        ]
        points = integer_points(points)
        area, perimeter, bbox = compute_polygon_metrics(points)
        polygon = PolygonData(
            id=self._next_polygon_id,
            points=points,
            is_hole=False,
            parent_id=None,
            category="via",
            shape_hint="box",
            area=area,
            perimeter=perimeter,
            bbox=bbox,
        )
        self.undo_stack.push(AddPolygonCommand(self, polygon, select_after_redo=True))
        return True

    def _via_rect_overlaps_existing(self, rect: QRectF) -> bool:
        for polygon in self._polygons.values():
            if not _is_via_polygon(polygon):
                continue
            if rect.intersects(_polygon_data_rect(polygon)):
                return True
        return False
