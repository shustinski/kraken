"""qt: frame matrix minimap."""
from __future__ import annotations
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QWidget,
)

class FrameMatrixMinimap(QWidget):
    """Compact matrix overview with viewport indication and click navigation."""

    WIDTH = 180
    HEIGHT = 120
    EDGE_MARGIN = 12
    CONTENT_MARGIN = 8

    def __init__(self, matrix_view: FrameMatrixView) -> None:
        super().__init__(matrix_view.viewport())
        self._matrix_view = matrix_view
        self.setObjectName("matrixMinimap")
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Нажмите или перетащите, чтобы перейти к нужной области")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        matrix_view.viewportChanged.connect(self._viewport_changed)
        self._reposition()
        self.show()
        self.raise_()

    def _viewport_changed(self, _visible_rect: QRectF) -> None:
        self._reposition()
        self.update()
        self.raise_()

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.move(
            max(0, parent.width() - self.width() - self.EDGE_MARGIN),
            max(0, parent.height() - self.height() - self.EDGE_MARGIN),
        )
        self.raise_()

    def _matrix_rect(self) -> QRectF:
        scene = self._matrix_view.sceneRect()
        content = QRectF(self.rect()).adjusted(
            self.CONTENT_MARGIN,
            self.CONTENT_MARGIN,
            -self.CONTENT_MARGIN,
            -self.CONTENT_MARGIN,
        )
        if scene.isEmpty():
            return content
        scale = min(content.width() / scene.width(), content.height() / scene.height())
        width = scene.width() * scale
        height = scene.height() * scale
        return QRectF(
            content.center().x() - width / 2,
            content.center().y() - height / 2,
            width,
            height,
        )

    def _map_scene_rect(self, scene_rect: QRectF) -> QRectF:
        scene = self._matrix_view.sceneRect()
        matrix = self._matrix_rect()
        if scene.isEmpty() or matrix.isEmpty():
            return QRectF()
        return QRectF(
            matrix.left() + (scene_rect.left() - scene.left()) / scene.width() * matrix.width(),
            matrix.top() + (scene_rect.top() - scene.top()) / scene.height() * matrix.height(),
            scene_rect.width() / scene.width() * matrix.width(),
            scene_rect.height() / scene.height() * matrix.height(),
        ).intersected(matrix)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.setBrush(QColor(11, 17, 32, 225))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)

        matrix = self._matrix_rect()
        painter.setPen(QPen(QColor("#64748b"), 1.0))
        painter.setBrush(QColor("#1e293b"))
        painter.drawRect(matrix)

        visible = self._map_scene_rect(self._matrix_view.visible_scene_rect())
        if not visible.isEmpty():
            painter.setPen(QPen(QColor("#60a5fa"), 1.5))
            painter.setBrush(QColor(37, 99, 235, 75))
            painter.drawRect(visible)
        painter.end()

    def _navigate(self, position: QPointF) -> None:
        matrix = self._matrix_rect()
        scene = self._matrix_view.sceneRect()
        if matrix.isEmpty() or scene.isEmpty():
            return
        x = min(matrix.right(), max(matrix.left(), position.x()))
        y = min(matrix.bottom(), max(matrix.top(), position.y()))
        scene_x = scene.left() + (x - matrix.left()) / matrix.width() * scene.width()
        scene_y = scene.top() + (y - matrix.top()) / matrix.height() * scene.height()
        self._matrix_view.centerOn(scene_x, scene_y)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._navigate(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._navigate(event.position())
            event.accept()
            return
        super().mouseMoveEvent(event)
