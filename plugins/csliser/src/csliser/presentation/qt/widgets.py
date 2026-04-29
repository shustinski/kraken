from __future__ import annotations

from PyQt6.QtCore import (
    QEasingCurve,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    pyqtProperty,
    pyqtSlot,
)
from PyQt6.QtGui import QBrush, QColor, QPaintEvent, QPainter, QPen
from PyQt6.QtWidgets import QCheckBox


class ExtensionCheckbox(QCheckBox):
    def __init__(self, folder: str = "", extension: str = "") -> None:
        super().__init__(extension)
        self.folder = folder
        self.extension = extension


class AnimatedToggle(QCheckBox):
    _transparent_pen = QPen(Qt.GlobalColor.transparent)
    _light_grey_pen = QPen(Qt.GlobalColor.lightGray)

    def __init__(
        self,
        parent=None,
        *,
        bar_color=Qt.GlobalColor.gray,
        checked_color="#3399ff",
        handle_color=Qt.GlobalColor.white,
    ) -> None:
        super().__init__(parent)
        self._bar_brush = QBrush(bar_color)
        self._bar_checked_brush = QBrush(QColor(checked_color).lighter())
        self._handle_brush = QBrush(handle_color)
        self._handle_checked_brush = QBrush(QColor(checked_color))
        self.setContentsMargins(8, 0, 8, 0)
        self._handle_position = 0.0
        self.animation = QPropertyAnimation(self, b"handle_position", self)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.animation.setDuration(180)
        self.stateChanged.connect(self.setup_animation)

    def sizeHint(self) -> QSize:
        return QSize(58, 36)

    def hitButton(self, pos: QPoint) -> bool:  # noqa: N802
        return self.contentsRect().contains(pos)

    @pyqtSlot(int)
    def setup_animation(self, value: int) -> None:
        self.animation.stop()
        self.animation.setEndValue(1.0 if value else 0.0)
        self.animation.start()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802, ARG002
        contents = self.contentsRect()
        handle_radius = round(0.24 * contents.height())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(self._transparent_pen)

        bar_rect = QRectF(0, 0, contents.width() - handle_radius, 0.40 * contents.height())
        bar_rect.moveCenter(QPointF(contents.center()))
        trail_length = contents.width() - 2 * handle_radius
        x_pos = contents.x() + handle_radius + trail_length * self._handle_position

        if self.isChecked():
            painter.setBrush(self._bar_checked_brush)
            painter.drawRoundedRect(bar_rect, bar_rect.height() / 2, bar_rect.height() / 2)
            painter.setBrush(self._handle_checked_brush)
        else:
            painter.setBrush(self._bar_brush)
            painter.drawRoundedRect(bar_rect, bar_rect.height() / 2, bar_rect.height() / 2)
            painter.setPen(self._light_grey_pen)
            painter.setBrush(self._handle_brush)

        painter.drawEllipse(QPointF(x_pos, bar_rect.center().y()), handle_radius, handle_radius)
        painter.end()

    @pyqtProperty(float)
    def handle_position(self) -> float:
        return self._handle_position

    @handle_position.setter
    def handle_position(self, position: float) -> None:
        self._handle_position = position
        self.update()
