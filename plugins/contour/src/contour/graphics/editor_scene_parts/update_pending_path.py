"""editor_scene: update pending path."""
from __future__ import annotations
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QPainterPath,
    QPen,
)
from contour.graphics.polygon_creation import (
    polygon_commit_acceptability,
)


class UpdatePendingPathMixin:
    def _update_pending_path(self) -> None:

        brush_width = self._pending_brush_width if self._pending_polyline_for_brush else float(self._pending_path_item.pen().widthF())

        if self._pending_polyline_for_brush:

            outline_color = QColor("#F7B801")

            if self._pending_points and brush_width >= 1.0:
                preview_points = list(self._pending_points)
                if self._pending_cursor is not None:
                    preview_points.append(self._pending_cursor)

                preview_color = QColor(outline_color)
                preview_color.setAlpha(96)
                preview_pen = QPen(
                    preview_color,
                    max(1.0, float(brush_width)),
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.BevelJoin,
                )
                preview_pen.setCosmetic(False)
                self._pending_path_item.setPen(preview_pen)
                self._pending_path_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                backbone = QPainterPath()
                if len(preview_points) == 1:
                    radius = max(1.0, float(brush_width)) / 2.0
                    x_coord, y_coord = preview_points[0]
                    backbone.addEllipse(
                        QRectF(
                            x_coord - radius,
                            y_coord - radius,
                            radius * 2.0,
                            radius * 2.0,
                        )
                    )
                    dot_color = QColor(outline_color)
                    dot_color.setAlpha(55)
                    self._pending_path_item.setBrush(QBrush(dot_color))
                    preview_pen.setWidthF(1.25)
                    preview_pen.setCosmetic(True)
                    self._pending_path_item.setPen(preview_pen)
                else:
                    backbone.moveTo(preview_points[0][0], preview_points[0][1])
                    for tail in preview_points[1:]:
                        backbone.lineTo(tail[0], tail[1])
                self._pending_path_item.setPath(backbone)

                return

            if len(self._pending_points) == 1 and self._pending_cursor is None:

                radius = max(1.0, brush_width) / 2.0

                hub_x, hub_y = self._pending_points[0]

                dot = QPainterPath()

                dot.addEllipse(QRectF(hub_x - radius, hub_y - radius, radius * 2.0, radius * 2.0))

                fill_color = QColor("#F7B801")

                fill_color.setAlpha(55)

                self._pending_path_item.setBrush(QBrush(fill_color))

                outline_pen = QPen(outline_color, 1.25)

                outline_pen.setCosmetic(True)

                self._pending_path_item.setPen(outline_pen)

                self._pending_path_item.setPath(dot)

                return

        preview_points = list(self._pending_points)
        if self._pending_cursor is not None:
            preview_points.append(self._pending_cursor)
        can_preview_closed = len(preview_points) >= 3
        preview_valid = False
        # Pointer motion only needs a visual continuation. Exact topology and
        # size checks run on clicks/commit; repeating them for every cursor
        # sample makes long vertex-authored polygons progressively slower.
        if can_preview_closed and self._pending_cursor is None:
            preview_valid, _reason = polygon_commit_acceptability(preview_points)
        preview_color = QColor(
            "#38BDF8"
            if preview_valid or (can_preview_closed and self._pending_cursor is not None)
            else "#EF4444"
            if can_preview_closed
            else "#F7B801"
        )

        dashed_pen_setup = QPen(preview_color, 1.5, Qt.PenStyle.DashLine)

        dashed_pen_setup.setCosmetic(True)

        dashed_pen_setup.setCapStyle(Qt.PenCapStyle.RoundCap)

        dashed_pen_setup.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

        dashed_pen_setup.setWidthF(self._pending_path_item.pen().widthF())

        if self._pending_polyline_for_brush:

            dashed_pen_setup.setCosmetic(False)

        self._pending_path_item.setPen(dashed_pen_setup)

        if can_preview_closed:
            fill_color = QColor(preview_color)
            fill_color.setAlpha(48)
            self._pending_path_item.setBrush(QBrush(fill_color))
        else:
            self._pending_path_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))

        backbone = QPainterPath()

        if preview_points:

            head = preview_points[0]

            backbone.moveTo(head[0], head[1])

            for tail in preview_points[1:]:
                backbone.lineTo(tail[0], tail[1])

            if can_preview_closed:
                backbone.closeSubpath()

        self._pending_path_item.setPath(backbone)
