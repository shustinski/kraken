"""Live hints of the filter settings: a small window that draws what a setting does, by its current
value, and plays while it is open.

A standard tool tip only shows text and still pictures; this one is a frameless window painted by
QPainter on a timer. It opens on hovering a setting (its title or slider) and stays open while the
slider is dragged, following the value. The animations run only while the window is open.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from PyQt6.QtCore import QEvent, QObject, QPoint, QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QGuiApplication, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QAbstractSlider, QWidget

from .ui_constants import GRID_INSPECTION_ERROR_TYPE_COLORS

BACKGROUND = QColor(22, 24, 28)
CELL = QColor(91, 97, 107)
CELL_LIT = QColor(214, 220, 228)
TEXT = QColor(205, 212, 222)
MUTED = QColor(130, 140, 152)
ORANGE = QColor(249, 115, 22)
WIDTH, HEIGHT = 300, 210
CANVAS_HEIGHT = 170

Values = dict
Scene = Callable[[QPainter, QRectF, float, Values], tuple[str, dict]]


def _color(kind: str) -> QColor:
    return QColor(GRID_INSPECTION_ERROR_TYPE_COLORS.get(kind, "#94a3b8"))


def _pen(color: QColor, width: float = 1.5, style=Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(color, width, style)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    return pen


# Scenes: each paints into ``rect`` at time ``t`` (seconds) and returns its caption (i18n key, args). ----


def paint_debris(p: QPainter, rect: QRectF, t: float, values: Values) -> str:
    limit = float(values.get("min_area", 24))
    pieces = [(0.14, 0.25, 2.2), (0.32, 0.62, 3.6), (0.52, 0.22, 1.6), (0.72, 0.55, 5.0), (0.88, 0.28, 2.9),
              (0.22, 0.82, 4.3), (0.62, 0.84, 2.0), (0.42, 0.42, 6.5), (0.86, 0.82, 3.3)]
    kept = 0
    for index, (fx, fy, r) in enumerate(pieces):
        area = int(round(math.pi * r * r * 2.0))
        centre = QPointF(rect.left() + fx * rect.width(), rect.top() + fy * rect.height())
        wobble = 1.0 + 0.07 * math.sin(t * 2.4 + index)
        radius = r * 1.9 * wobble
        keep = area >= limit
        kept += keep
        color = _color("small_artifact") if keep else QColor(MUTED)
        if not keep:
            color.setAlpha(int(70 + 50 * (0.5 + 0.5 * math.sin(t * 3 + index))))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawEllipse(centre, radius, radius * 0.8)
        if keep:
            p.setPen(_pen(_color("small_artifact"), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(centre, radius + 6, radius + 6)
        p.setPen(TEXT if keep else MUTED)
        p.drawText(QPointF(centre.x() + radius + 4, centre.y() + 4), f"{area}")
    return "live_hint.debris", {"limit": int(limit), "kept": kept, "total": len(pieces)}


def paint_sparse(p: QPainter, rect: QRectF, t: float, values: Values) -> str:
    reach_x = float(values.get("reach_x", 4))
    reach_y = float(values.get("reach_y", 4))
    need = int(values.get("free_sides", 3))
    pitch_x, pitch_y, cell_w, cell_h = 22.0, 30.0, 12.0, 20.0
    cols = int(rect.width() // pitch_x)
    rows = int(rect.height() // pitch_y)
    centre = (cols // 2, rows // 2)
    # One neighbour walks along the diagonal towards the cell and back: inside the area it takes two
    # sides at once (top and right); a row neighbour further left is outside a small area.
    cycle = (1, 2, 3, 4, 5, 6, 5, 4, 3, 2)
    walk = cycle[int(t * 0.8) % len(cycle)]
    neighbours = {(centre[0] + walk, centre[1] - max(1, walk // 2)), (centre[0] - 5, centre[1]), (0, rows - 1), (cols - 1, rows - 1)}

    def box(col: int, row: int) -> QRectF:
        return QRectF(rect.left() + 6 + col * pitch_x, rect.top() + 4 + row * pitch_y, cell_w, cell_h)

    target = box(*centre)
    area = target.adjusted(-reach_x * pitch_x, -reach_y * pitch_y, reach_x * pitch_x, reach_y * pitch_y)
    pulse = 0.5 + 0.5 * math.sin(t * 3.0)
    fill = QColor(_color("sparse_cell"))
    fill.setAlpha(int(18 + 22 * pulse))
    p.fillRect(area.intersected(rect), fill)
    p.setPen(_pen(_color("sparse_cell"), 1.0, Qt.PenStyle.DashLine))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(area.intersected(rect))
    sides = {"left": False, "right": False, "top": False, "bottom": False}
    for col, row in neighbours:
        cell = box(col, row)
        inside = area.contains(cell.center())
        if inside:
            if cell.center().x() < target.left():
                sides["left"] = True
            if cell.center().x() > target.right():
                sides["right"] = True
            if cell.center().y() < target.top():
                sides["top"] = True
            if cell.center().y() > target.bottom():
                sides["bottom"] = True
        p.fillRect(cell, CELL_LIT if inside else CELL)
    for name, (a, b) in {
        "left": (QPointF(target.left() - 6, target.center().y()), QPointF(target.left() - 2, target.center().y())),
        "right": (QPointF(target.right() + 2, target.center().y()), QPointF(target.right() + 6, target.center().y())),
        "top": (QPointF(target.center().x(), target.top() - 6), QPointF(target.center().x(), target.top() - 2)),
        "bottom": (QPointF(target.center().x(), target.bottom() + 2), QPointF(target.center().x(), target.bottom() + 6)),
    }.items():
        p.setPen(_pen(ORANGE if sides[name] else QColor(58, 64, 74), 3.0))
        p.drawLine(a, b)
    free = 4 - sum(sides.values())
    sparse = free >= need
    p.fillRect(target, QColor(140, 150, 162))
    if sparse:
        p.setPen(_pen(_color("sparse_cell"), 2.0))
        gap, arm = 3.0, 5.0
        outer = target.adjusted(-gap, -gap, gap, gap)
        for corner, dx, dy in ((outer.topLeft(), 1, 1), (outer.topRight(), -1, 1), (outer.bottomLeft(), 1, -1), (outer.bottomRight(), -1, -1)):
            p.drawLine(corner, QPointF(corner.x() + dx * arm, corner.y()))
            p.drawLine(corner, QPointF(corner.x(), corner.y() + dy * arm))
    return "live_hint.sparse_yes" if sparse else "live_hint.sparse_no", {"x": f"{reach_x:g}", "y": f"{reach_y:g}", "free": free, "need": need}


def paint_conflict(p: QPainter, rect: QRectF, t: float, values: Values) -> str:
    min_area = float(values.get("min_area", 6))
    min_share = float(values.get("min_share", 10)) / 100.0
    cell = QRectF(rect.center().x() - 30, rect.top() + 25, 60, 100)
    p.fillRect(cell, CELL)
    # The other network's blob slides over the cell and back.
    phase = 0.5 + 0.5 * math.sin(t * 1.2)
    blob_w = 70.0
    blob_x = cell.right() - phase * (cell.width() * 0.9)
    blob = QRectF(blob_x, cell.top() + 30, blob_w, 45)
    overlap = cell.intersected(blob)
    blob_color = QColor(56, 189, 248, 90)
    p.fillRect(blob, blob_color)
    scale_px = 0.25
    overlap_px = overlap.width() * overlap.height() * scale_px if not overlap.isEmpty() else 0.0
    share = (overlap.width() * overlap.height()) / (cell.width() * cell.height()) if not overlap.isEmpty() else 0.0
    conflict = overlap_px >= min_area and share >= min_share
    if not overlap.isEmpty():
        color = _color("class_conflict") if conflict else QColor(MUTED)
        p.save()
        p.setClipRect(overlap)
        p.setPen(_pen(color, 1.5))
        step = 6.0
        offset = -overlap.height()
        while offset < overlap.width():
            p.drawLine(QPointF(overlap.left() + offset, overlap.bottom()), QPointF(overlap.left() + offset + overlap.height(), overlap.top()))
            offset += step
        p.restore()
        p.setPen(_pen(color, 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(overlap)
    p.setPen(MUTED)
    p.drawText(QPointF(cell.left() - 30, cell.top() - 6), "1")
    p.drawText(QPointF(blob.right() - 10, blob.bottom() + 14), "0")
    args = {"area": f"{overlap_px:.0f}", "share": f"{share * 100:.0f}", "min_area": f"{min_area:.0f}", "min_share": f"{min_share * 100:.0f}"}
    return ("live_hint.conflict_yes" if conflict else "live_hint.conflict_no"), args


def paint_agreement(p: QPainter, rect: QRectF, t: float, values: Values) -> str:
    limit = float(values.get("cover", 30)) / 100.0
    cell = QRectF(rect.center().x() - 30, rect.top() + 18, 60, 96)
    cover = 0.5 + 0.5 * math.sin(t * 0.9)
    found_by_both = cover >= limit
    p.fillRect(cell, CELL)
    covered = QRectF(cell.left(), cell.bottom() - cell.height() * cover, cell.width(), cell.height() * cover)
    p.fillRect(covered, QColor(148, 163, 184, 120))
    color = QColor(MUTED) if found_by_both else _color("network_only")
    p.setPen(_pen(color, 2.0, Qt.PenStyle.DotLine))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(cell.adjusted(-2, -2, 2, 2))
    bar = QRectF(rect.left() + 20, rect.bottom() - 26, rect.width() - 40, 8)
    p.fillRect(bar, QColor(45, 50, 58))
    p.fillRect(QRectF(bar.left(), bar.top(), bar.width() * cover, bar.height()), QColor(148, 163, 184))
    marker = bar.left() + bar.width() * limit
    p.setPen(_pen(_color("network_only"), 2.0))
    p.drawLine(QPointF(marker, bar.top() - 5), QPointF(marker, bar.bottom() + 5))
    args = {"cover": f"{cover * 100:.0f}", "limit": f"{limit * 100:.0f}"}
    return ("live_hint.agreement_both" if found_by_both else "live_hint.agreement_one"), args


SCENES: dict[str, Scene] = {
    "debris": paint_debris,
    "sparse": paint_sparse,
    "conflict": paint_conflict,
    "agreement": paint_agreement,
}


class HintPopup(QWidget):
    """The window of one hint: a picture that plays and its caption."""

    def __init__(self, scene: Scene, values: Callable[[], Values], translate: Callable[..., str], parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self._t = translate
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFixedSize(WIDTH, HEIGHT)
        self._scene = scene
        self._values = values
        self._time = 0.0
        self.caption = ""
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt)
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt)
        self._timer.stop()
        super().hideEvent(event)

    def _tick(self) -> None:
        self._time += self._timer.interval() / 1000.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), BACKGROUND)
        p.setPen(_pen(QColor(58, 64, 74), 1.0))
        p.drawRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5))
        canvas = QRectF(8, 8, WIDTH - 16, CANVAS_HEIGHT - 16)
        font = QFont(p.font())
        font.setPointSize(8)
        p.setFont(font)
        p.save()
        p.setClipRect(canvas)
        key, args = self._scene(p, canvas, self._time, dict(self._values() or {}))
        self.caption = self._t(key, **args)
        p.restore()
        p.setPen(TEXT)
        p.drawText(QRectF(10, CANVAS_HEIGHT - 6, WIDTH - 20, HEIGHT - CANVAS_HEIGHT), Qt.TextFlag.TextWordWrap, self.caption)
        p.end()

    def show_beside(self, anchor: QWidget) -> None:
        """Left of the anchor (the panel is on the right of the window), else right of it."""

        top_left = anchor.mapToGlobal(QPoint(0, 0))
        screen = QGuiApplication.screenAt(top_left) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry() if screen is not None else None
        x = top_left.x() - WIDTH - 10
        if area is not None and x < area.left():
            x = top_left.x() + anchor.width() + 10
        y = top_left.y() - HEIGHT // 3
        if area is not None:
            y = max(area.top(), min(y, area.bottom() - HEIGHT))
        self.move(x, y)
        self.show()
        self.raise_()


class LiveHint(QObject):
    """Opens a hint on hovering its widgets, keeps it while a slider among them is dragged."""

    def __init__(self, scene_key: str, values: Callable[[], Values], widgets, *, enabled: Callable[[], bool], translate: Callable[..., str], parent: QWidget) -> None:
        super().__init__(parent)
        self.popup = HintPopup(SCENES[scene_key], values, translate, parent)
        self._widgets = [widget for widget in widgets if widget is not None]
        self._enabled = enabled
        self._anchor = parent
        self._open_timer = QTimer(self)
        self._open_timer.setSingleShot(True)
        self._open_timer.setInterval(450)
        self._open_timer.timeout.connect(self._open)
        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.setInterval(350)
        self._close_timer.timeout.connect(self._close_if_idle)
        for widget in self._widgets:
            widget.installEventFilter(self)
            if isinstance(widget, QAbstractSlider):
                widget.sliderPressed.connect(self._open)
                widget.sliderReleased.connect(self._close_timer.start)

    def _dragging(self) -> bool:
        return any(isinstance(widget, QAbstractSlider) and widget.isSliderDown() for widget in self._widgets)

    def _hovered(self) -> bool:
        return any(widget.underMouse() for widget in self._widgets)

    def _open(self) -> None:
        if not self._enabled() or not any(widget.isVisible() for widget in self._widgets):
            return
        self._close_timer.stop()
        if not self.popup.isVisible():
            self.popup.show_beside(self._anchor)

    def _close_if_idle(self) -> None:
        if self._dragging() or self._hovered():
            return
        self.popup.hide()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        kind = event.type()
        if kind == QEvent.Type.Enter:
            self._open_timer.start()
        elif kind == QEvent.Type.Leave:
            self._open_timer.stop()
            self._close_timer.start()
        elif kind in (QEvent.Type.Hide, QEvent.Type.WindowDeactivate):
            self._open_timer.stop()
            self.popup.hide()
        return super().eventFilter(watched, event)
