"""details_dialog: point matches pixmap."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap


class PointMatchesPixmapMixin:
    def _point_matches_pixmap(self, first_key: str | None, second_key: str | None) -> QPixmap:
        cache_key = (
            "point_matches",
            first_key,
            second_key,
            float(self._payload.get("point_match_radius") or 3.0),
            self._selection_cache_key(),
            self._named_color("difference_mask").name(QColor.NameFormat.HexArgb),
            self._named_color("first_mask").name(QColor.NameFormat.HexArgb),
            self._named_color("second_mask").name(QColor.NameFormat.HexArgb),
        )
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        base = self._base_array()
        height, width = base.shape
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        first_view = self._source_point_view(first_key)
        second_view = self._source_point_view(second_key)
        first_points = tuple(getattr(first_view, 'points', ())) if first_view is not None else tuple()
        second_points = tuple(getattr(second_view, 'points', ())) if second_view is not None else tuple()
        tile_rect = self._selection_crop_rect()
        x_offset = 0.0
        y_offset = 0.0
        if tile_rect is not None:
            x0 = float(tile_rect.left())
            y0 = float(tile_rect.top())
            x1 = float(tile_rect.right())
            y1 = float(tile_rect.bottom())
            x_offset = x0
            y_offset = y0
            first_points = tuple(
                point for point in first_points
                if x0 <= float(getattr(point, 'x', 0.0)) <= x1 and y0 <= float(getattr(point, 'y', 0.0)) <= y1
            )
            second_points = tuple(
                point for point in second_points
                if x0 <= float(getattr(point, 'x', 0.0)) <= x1 and y0 <= float(getattr(point, 'y', 0.0)) <= y1
            )
        pairs, matched_first, matched_second = self._point_match_pairs(first_points, second_points, float(self._payload.get('point_match_radius') or 3.0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        line_pen = QPen(self._named_color("difference_mask"), 1.5)
        painter.setPen(line_pen)
        for index_a, index_b, _distance in pairs:
            point_a = first_points[index_a]
            point_b = second_points[index_b]
            painter.drawLine(
                int(round(float(getattr(point_a, 'x', 0.0)) - x_offset)),
                int(round(float(getattr(point_a, 'y', 0.0)) - y_offset)),
                int(round(float(getattr(point_b, 'x', 0.0)) - x_offset)),
                int(round(float(getattr(point_b, 'y', 0.0)) - y_offset)),
            )
        for index, point in enumerate(first_points):
            radius = max(2, int(round(float(getattr(point, 'radius', 0.0)) + 1.0)))
            color = QColor(70, 220, 120, 240) if index in matched_first else self._named_color("first_mask")
            painter.setPen(QPen(color, 1.2))
            painter.setBrush(color)
            x = int(round(float(getattr(point, 'x', 0.0)) - x_offset))
            y = int(round(float(getattr(point, 'y', 0.0)) - y_offset))
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
        for index, point in enumerate(second_points):
            if index in matched_second:
                continue
            radius = max(2, int(round(float(getattr(point, 'radius', 0.0)) + 1.0)))
            color = self._named_color("second_mask")
            painter.setPen(QPen(color, 1.2))
            painter.setBrush(color)
            x = int(round(float(getattr(point, 'x', 0.0)) - x_offset))
            y = int(round(float(getattr(point, 'y', 0.0)) - y_offset))
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
        painter.end()
        self._overlay_cache[cache_key] = pixmap
        return pixmap
