"""Outline icons of the run toolbar, drawn in code: one look under every Qt style.

Every icon has a normal and a disabled (dim) state, so it is clear what can be pressed.
"""

from __future__ import annotations

from functools import lru_cache

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

ICON_COLOR = "#d5e1ee"
DISABLED_COLOR = "#4f5b68"
_SIZE = 64


def _pen(color: str) -> QPen:
    pen = QPen(QColor(color), 5.0)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _draw(name: str, painter: QPainter, color: str) -> None:
    painter.setPen(_pen(color))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    if name == "folder_add":
        path = QPainterPath()
        path.moveTo(8, 18)
        path.lineTo(24, 18)
        path.lineTo(29, 24)
        path.lineTo(56, 24)
        path.lineTo(56, 52)
        path.lineTo(8, 52)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawLine(QPointF(32, 31), QPointF(32, 45))
        painter.drawLine(QPointF(25, 38), QPointF(39, 38))
    elif name == "clear":
        painter.drawLine(QPointF(18, 18), QPointF(46, 46))
        painter.drawLine(QPointF(46, 18), QPointF(18, 46))
    elif name == "export":
        painter.drawLine(QPointF(32, 10), QPointF(32, 40))
        painter.drawPolyline([QPointF(20, 29), QPointF(32, 41), QPointF(44, 29)])
        painter.drawPolyline([QPointF(12, 42), QPointF(12, 54), QPointF(52, 54), QPointF(52, 42)])
    elif name == "search":
        painter.drawEllipse(QRectF(12, 12, 28, 28))
        painter.drawLine(QPointF(37, 37), QPointF(52, 52))
    elif name == "play":
        painter.setBrush(QColor(color))
        path = QPainterPath()
        path.moveTo(20, 14)
        path.lineTo(50, 32)
        path.lineTo(20, 50)
        path.closeSubpath()
        painter.drawPath(path)
    elif name == "stop":
        painter.setBrush(QColor(color))
        painter.drawRoundedRect(QRectF(18, 18, 28, 28), 3, 3)
    elif name == "pause":
        painter.setBrush(QColor(color))
        painter.drawRoundedRect(QRectF(18, 16, 9, 32), 2, 2)
        painter.drawRoundedRect(QRectF(37, 16, 9, 32), 2, 2)
    elif name == "grip":
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        for x in (24, 40):
            for y in (16, 32, 48):
                painter.drawEllipse(QPointF(x, y), 4.0, 4.0)
    elif name == "settings":
        for y, knob in ((18, 40), (32, 22), (46, 34)):
            painter.drawLine(QPointF(12, y), QPointF(52, y))
            painter.setBrush(QColor("#11161d"))
            painter.drawEllipse(QPointF(knob, y), 5.5, 5.5)
            painter.setBrush(Qt.BrushStyle.NoBrush)
    elif name == "trash":
        painter.drawLine(QPointF(12, 18), QPointF(52, 18))
        painter.drawPolyline([QPointF(25, 18), QPointF(27, 11), QPointF(37, 11), QPointF(39, 18)])
        path = QPainterPath()
        path.moveTo(17, 18)
        path.lineTo(20, 54)
        path.lineTo(44, 54)
        path.lineTo(47, 18)
        painter.drawPath(path)
        painter.drawLine(QPointF(28, 27), QPointF(28, 45))
        painter.drawLine(QPointF(36, 27), QPointF(36, 45))
    elif name == "chevron_right":
        painter.drawPolyline([QPointF(24, 14), QPointF(42, 32), QPointF(24, 50)])
    elif name == "chevron_down":
        painter.drawPolyline([QPointF(14, 24), QPointF(32, 42), QPointF(50, 24)])
    elif name in ("sidebar_collapse", "sidebar_expand"):
        painter.drawRoundedRect(QRectF(8, 12, 48, 40), 6, 6)
        painter.drawLine(QPointF(24, 12), QPointF(24, 52))
        if name == "sidebar_collapse":
            painter.drawPolyline([QPointF(42, 24), QPointF(34, 32), QPointF(42, 40)])
        else:
            painter.drawPolyline([QPointF(36, 24), QPointF(44, 32), QPointF(36, 40)])
    elif name == "refresh":
        painter.drawArc(QRectF(14, 14, 36, 36), 60 * 16, 270 * 16)
        painter.setBrush(QColor(color))
        path = QPainterPath()
        path.moveTo(41, 8)
        path.lineTo(52, 18)
        path.lineTo(38, 22)
        path.closeSubpath()
        painter.drawPath(path)


def _pixmap(name: str, color: str) -> QPixmap:
    pixmap = QPixmap(_SIZE, _SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    _draw(name, painter, color)
    painter.end()
    return pixmap


def save_icon_png(name: str, path, size: int = 24, color: str = ICON_COLOR) -> None:
    """Write one icon as PNG (style sheet images such as the section chevrons)."""

    _pixmap(name, color).scaled(
        size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    ).save(str(path))


@lru_cache(maxsize=None)
def toolbar_icon(name: str, color: str = ICON_COLOR) -> QIcon:
    """Toolbar: ``folder_add``, ``clear``, ``export``, ``search``, ``play``, ``stop``, ``pause``, ``refresh``.

    Layer rows: ``grip``, ``settings``, ``trash``. Sections and the side panel: ``chevron_right``,
    ``chevron_down``, ``sidebar_collapse``, ``sidebar_expand``.
    """

    icon = QIcon()
    icon.addPixmap(_pixmap(name, color), QIcon.Mode.Normal)
    icon.addPixmap(_pixmap(name, DISABLED_COLOR), QIcon.Mode.Disabled)
    return icon
