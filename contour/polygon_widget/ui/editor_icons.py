"""Pure helpers for editor toolbar/tool icons.

All functions here are stateless — they build ``QIcon`` / ``QPixmap`` instances
from primitives. Extracted from :mod:`polygon_widget.widget` to keep the widget
class focused on orchestration rather than low-level painting.
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

from ..graphics.tools import EditorTool

TOOLBAR_ICON_SIZE_PX = 28
TOOLBAR_BUTTON_SIZE_PX = 34
TOOLBAR_ICON_CANVAS_SIZE_PX = 72


def _draw_vertex_marker(painter: QPainter, point: QPointF, stroke: QColor, fill: QColor, radius: float) -> None:
    painter.setPen(QPen(stroke, 1.2))
    painter.setBrush(QBrush(fill))
    painter.drawEllipse(QRectF(point.x() - radius, point.y() - radius, radius * 2.0, radius * 2.0))


def _draw_badge(painter: QPainter, center: QPointF, color: QColor, symbol: str) -> None:
    badge_rect = QRectF(center.x() - 4.3, center.y() - 4.3, 8.6, 8.6)
    painter.setPen(QPen(color.darker(120), 1.0))
    painter.setBrush(QBrush(color))
    painter.drawEllipse(badge_rect)
    painter.setPen(QPen(QColor("#FFFFFF"), 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    if symbol == "+":
        painter.drawLine(QPointF(center.x() - 2.2, center.y()), QPointF(center.x() + 2.2, center.y()))
        painter.drawLine(QPointF(center.x(), center.y() - 2.2), QPointF(center.x(), center.y() + 2.2))
    elif symbol == "-":
        painter.drawLine(QPointF(center.x() - 2.2, center.y()), QPointF(center.x() + 2.2, center.y()))
    else:
        painter.drawLine(
            QPointF(center.x() - 1.9, center.y() - 1.9),
            QPointF(center.x() + 1.9, center.y() + 1.9),
        )
        painter.drawLine(
            QPointF(center.x() - 1.9, center.y() + 1.9),
            QPointF(center.x() + 1.9, center.y() - 1.9),
        )


def _draw_arrow_head(painter: QPainter, base: QPointF, tip: QPointF) -> None:
    vector_x = tip.x() - base.x()
    vector_y = tip.y() - base.y()
    if abs(vector_x) >= abs(vector_y):
        direction = 1.0 if vector_x >= 0 else -1.0
        left = QPointF(base.x() + 1.6 * direction, base.y() - 1.4)
        right = QPointF(base.x() + 1.6 * direction, base.y() + 1.4)
    else:
        direction = 1.0 if vector_y >= 0 else -1.0
        left = QPointF(base.x() - 1.4, base.y() + 1.6 * direction)
        right = QPointF(base.x() + 1.4, base.y() + 1.6 * direction)
    painter.drawLine(base, left)
    painter.drawLine(base, right)


def _paint_select_icon(painter: QPainter, stroke: QColor) -> None:
    path = QPainterPath()
    path.moveTo(5.5, 4.0)
    path.lineTo(5.5, 21.0)
    path.lineTo(10.0, 16.5)
    path.lineTo(12.8, 23.0)
    path.lineTo(16.0, 21.8)
    path.lineTo(13.2, 15.6)
    path.lineTo(20.8, 15.6)
    path.closeSubpath()
    painter.setPen(QPen(stroke, 1.9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.fillPath(path, QBrush(QColor("#FFFFFF")))
    painter.drawPath(path)


def _paint_select_area_icon(painter: QPainter, stroke: QColor, accent: QColor) -> None:
    painter.setPen(QPen(accent, 1.8, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
    painter.drawRect(QRectF(5.0, 6.0, 16.0, 13.0))
    _paint_select_icon(painter, stroke)


def _paint_pan_icon(painter: QPainter, stroke: QColor, accent: QColor) -> None:
    center = QPointF(14.0, 14.0)
    painter.setPen(QPen(stroke, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawLine(QPointF(14.0, 5.0), QPointF(14.0, 23.0))
    painter.drawLine(QPointF(5.0, 14.0), QPointF(23.0, 14.0))
    _draw_arrow_head(painter, QPointF(14.0, 5.0), QPointF(14.0, 2.0))
    _draw_arrow_head(painter, QPointF(14.0, 23.0), QPointF(14.0, 26.0))
    _draw_arrow_head(painter, QPointF(5.0, 14.0), QPointF(2.0, 14.0))
    _draw_arrow_head(painter, QPointF(23.0, 14.0), QPointF(26.0, 14.0))
    painter.setPen(QPen(accent, 1.8))
    painter.setBrush(QBrush(accent))
    painter.drawEllipse(QRectF(center.x() - 2.2, center.y() - 2.2, 4.4, 4.4))


def _paint_ruler_icon(painter: QPainter, stroke: QColor, accent: QColor) -> None:
    start = QPointF(5.0, 19.5)
    end = QPointF(23.0, 8.5)
    painter.setPen(QPen(stroke, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawLine(start, end)
    painter.setPen(QPen(accent, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    for tick in range(1, 5):
        base_x = 6.5 + tick * 3.4
        base_y = 18.6 - tick * 2.1
        painter.drawLine(QPointF(base_x, base_y), QPointF(base_x - 1.0, base_y - 1.8))
    _draw_vertex_marker(painter, start, stroke, QColor("#FFFFFF"), radius=1.6)
    _draw_vertex_marker(painter, end, stroke, QColor("#FFFFFF"), radius=1.6)


def _paint_polygon_badge_icon(
    painter: QPainter,
    stroke: QColor,
    badge_color: QColor,
    badge_symbol: str,
) -> None:
    points = [
        QPointF(4.5, 18.5),
        QPointF(8.6, 7.0),
        QPointF(18.0, 8.6),
        QPointF(20.0, 18.0),
        QPointF(12.0, 22.0),
    ]
    polygon = QPolygonF(points)
    painter.setPen(QPen(stroke, 2.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPolygon(polygon)
    for point in points:
        _draw_vertex_marker(painter, point, stroke, QColor("#FFFFFF"), radius=1.9)
    _draw_badge(painter, QPointF(20.5, 6.5), badge_color, badge_symbol)


def _paint_vertex_edit_icon(
    painter: QPainter,
    stroke: QColor,
    neutral: QColor,
    badge_color: QColor,
    badge_symbol: str,
) -> None:
    polyline = [QPointF(4.5, 18.0), QPointF(11.0, 8.0), QPointF(19.0, 18.2)]
    painter.setPen(QPen(stroke, 2.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.drawPolyline(QPolygonF(polyline))
    _draw_vertex_marker(painter, polyline[0], stroke, QColor("#FFFFFF"), radius=1.8)
    _draw_vertex_marker(painter, polyline[2], stroke, QColor("#FFFFFF"), radius=1.8)
    _draw_vertex_marker(painter, polyline[1], stroke, neutral, radius=2.4)
    _draw_badge(painter, QPointF(20.0, 6.5), badge_color, badge_symbol)


def _paint_move_vertex_icon(painter: QPainter, stroke: QColor, accent: QColor) -> None:
    polygon = QPolygonF(
        [
            QPointF(4.5, 18.4),
            QPointF(8.8, 8.0),
            QPointF(17.0, 9.0),
            QPointF(19.6, 17.5),
            QPointF(11.4, 21.2),
        ]
    )
    painter.setPen(QPen(stroke, 2.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPolygon(polygon)
    target = QPointF(17.0, 9.0)
    _draw_vertex_marker(painter, target, stroke, accent, radius=2.5)
    painter.setPen(QPen(accent, 1.9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawLine(QPointF(17.0, 4.2), QPointF(17.0, 13.8))
    painter.drawLine(QPointF(12.2, 9.0), QPointF(21.8, 9.0))
    _draw_arrow_head(painter, QPointF(17.0, 4.2), QPointF(17.0, 1.6))
    _draw_arrow_head(painter, QPointF(17.0, 13.8), QPointF(17.0, 16.4))
    _draw_arrow_head(painter, QPointF(12.2, 9.0), QPointF(9.6, 9.0))
    _draw_arrow_head(painter, QPointF(21.8, 9.0), QPointF(24.4, 9.0))


def _paint_brush_icon(painter: QPainter, stroke: QColor, accent: QColor) -> None:
    painter.setPen(QPen(stroke, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    path = QPainterPath()
    path.moveTo(7.0, 20.5)
    path.cubicTo(9.0, 15.0, 13.0, 10.0, 18.0, 6.5)
    path.lineTo(21.0, 9.5)
    path.cubicTo(17.5, 14.5, 12.5, 18.5, 7.0, 20.5)
    painter.drawPath(path)
    painter.setBrush(QBrush(accent))
    painter.setPen(QPen(accent, 1.0))
    painter.drawEllipse(QRectF(18.8, 5.2, 4.6, 4.6))


def _paint_via_icon(painter: QPainter, stroke: QColor, accent: QColor) -> None:
    painter.setPen(QPen(stroke, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(QRectF(6.0, 7.0, 16.0, 14.0))
    painter.setPen(QPen(accent, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.setBrush(QBrush(accent))
    painter.drawEllipse(QRectF(10.0, 9.0, 8.0, 8.0))
    painter.drawLine(QPointF(14.0, 4.5), QPointF(14.0, 7.0))
    painter.drawLine(QPointF(14.0, 21.0), QPointF(14.0, 23.5))
    painter.drawLine(QPointF(3.5, 14.0), QPointF(6.0, 14.0))
    painter.drawLine(QPointF(22.0, 14.0), QPointF(24.5, 14.0))


def _paint_history_icon(painter: QPainter, stroke: QColor, mirrored: bool) -> None:
    painter.setPen(QPen(stroke, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    path = QPainterPath()
    if mirrored:
        path.moveTo(7.0, 8.0)
        path.cubicTo(13.0, 3.5, 22.5, 6.0, 22.0, 14.0)
        path.cubicTo(21.5, 21.0, 13.5, 23.5, 8.5, 20.0)
        painter.drawPath(path)
        _draw_arrow_head(painter, QPointF(7.2, 8.0), QPointF(3.8, 9.0))
    else:
        path.moveTo(21.0, 8.0)
        path.cubicTo(15.0, 3.5, 5.5, 6.0, 6.0, 14.0)
        path.cubicTo(6.5, 21.0, 14.5, 23.5, 19.5, 20.0)
        painter.drawPath(path)
        _draw_arrow_head(painter, QPointF(20.8, 8.0), QPointF(24.2, 9.0))


def _paint_zoom_icon(painter: QPainter, stroke: QColor, accent: QColor, *, add: bool) -> None:
    painter.setPen(QPen(stroke, 2.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(QRectF(5.0, 5.0, 12.0, 12.0))
    painter.drawLine(QPointF(15.2, 15.2), QPointF(22.8, 22.8))
    painter.setPen(QPen(accent, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawLine(QPointF(8.5, 11.0), QPointF(13.5, 11.0))
    if add:
        painter.drawLine(QPointF(11.0, 8.5), QPointF(11.0, 13.5))


def _paint_fit_icon(painter: QPainter, stroke: QColor, accent: QColor) -> None:
    painter.setPen(QPen(stroke, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawRect(QRectF(7.0, 7.0, 14.0, 14.0))
    painter.setPen(QPen(accent, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawLine(QPointF(5.0, 10.0), QPointF(9.0, 10.0))
    painter.drawLine(QPointF(10.0, 5.0), QPointF(10.0, 9.0))
    painter.drawLine(QPointF(19.0, 5.0), QPointF(19.0, 9.0))
    painter.drawLine(QPointF(19.0, 19.0), QPointF(19.0, 23.0))
    painter.drawLine(QPointF(5.0, 19.0), QPointF(9.0, 19.0))
    painter.drawLine(QPointF(19.0, 19.0), QPointF(23.0, 19.0))


def _new_painter() -> tuple[QPixmap, QPainter]:
    canvas_size = TOOLBAR_ICON_CANVAS_SIZE_PX
    pixmap = QPixmap(canvas_size, canvas_size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    scale_factor = canvas_size / 28.0
    painter.scale(scale_factor, scale_factor)
    return pixmap, painter


def create_editor_tool_icon(tool: EditorTool) -> QIcon:
    pixmap, painter = _new_painter()
    stroke = QColor("#FFFFFF")
    neutral = QColor("#E2E8F0")
    accent = QColor("#38BDF8")
    success = QColor("#4ADE80")
    warning = QColor("#FDBA74")
    danger = QColor("#FB7185")

    if tool == EditorTool.SELECT:
        _paint_select_icon(painter, stroke)
    elif tool == EditorTool.SELECT_AREA:
        _paint_select_area_icon(painter, stroke, accent)
    elif tool == EditorTool.PAN:
        _paint_pan_icon(painter, stroke, accent)
    elif tool == EditorTool.RULER:
        _paint_ruler_icon(painter, stroke, warning)
    elif tool == EditorTool.ADD_POLYGON:
        _paint_polygon_badge_icon(painter, stroke, accent, "+")
    elif tool == EditorTool.BRUSH:
        _paint_brush_icon(painter, stroke, success)
    elif tool == EditorTool.ADD_VIA:
        _paint_via_icon(painter, stroke, QColor("#A78BFA"))
    elif tool == EditorTool.ADD_VERTEX:
        _paint_vertex_edit_icon(painter, stroke, neutral, success, "+")
    elif tool == EditorTool.DELETE_VERTEX:
        _paint_vertex_edit_icon(painter, stroke, neutral, danger, "-")
    elif tool == EditorTool.MOVE_VERTEX:
        _paint_move_vertex_icon(painter, stroke, warning)
    elif tool == EditorTool.DELETE_POLYGON:
        _paint_polygon_badge_icon(painter, stroke, danger, "x")
    painter.end()
    return QIcon(pixmap)


def create_editor_action_icon(action: str) -> QIcon:
    pixmap, painter = _new_painter()
    stroke = QColor("#FFFFFF")
    accent = QColor("#38BDF8")

    if action == "undo":
        _paint_history_icon(painter, stroke, mirrored=False)
    elif action == "redo":
        _paint_history_icon(painter, stroke, mirrored=True)
    elif action == "zoom_in":
        _paint_zoom_icon(painter, stroke, accent, add=True)
    elif action == "zoom_out":
        _paint_zoom_icon(painter, stroke, accent, add=False)
    else:
        _paint_fit_icon(painter, stroke, accent)
    painter.end()
    return QIcon(pixmap)


__all__ = [
    "TOOLBAR_BUTTON_SIZE_PX",
    "TOOLBAR_ICON_CANVAS_SIZE_PX",
    "TOOLBAR_ICON_SIZE_PX",
    "create_editor_action_icon",
    "create_editor_tool_icon",
]
