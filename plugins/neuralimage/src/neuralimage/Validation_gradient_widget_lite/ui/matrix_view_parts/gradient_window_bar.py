"""matrix_view: gradient window bar."""
from __future__ import annotations
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import (
    QWidget,
)
from neuralimage.Validation_gradient_widget_lite.ui.ui_constants import (
    DEFAULT_ERROR_WINDOW,
    DEFAULT_GRADIENT_NAME,
    GRADIENT_PRESETS,
    GRADIENT_RANGE_SELECTOR_MIN_HEIGHT,
    PANEL_TEXT,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.compute_auto_color_window import interpolate_gradient_color

class _GradientWindowBar(QWidget):
    """Render and edit the active error window on top of the current gradient."""

    rangeEdited = pyqtSignal(float, float)

    def __init__(self, parent=None) -> None:
        """Initialize the interactive error-window bar."""
        super().__init__(parent)
        self._gradient_name = DEFAULT_GRADIENT_NAME
        self._low_bound, self._high_bound = DEFAULT_ERROR_WINDOW
        self._active_handle: str | None = None
        self.setMinimumHeight(max(26, GRADIENT_RANGE_SELECTOR_MIN_HEIGHT // 2))
        self.setMouseTracking(True)

    def set_gradient_name(self, gradient_name: str) -> None:
        """Set the gradient used by the range bar."""
        self._gradient_name = gradient_name if gradient_name in GRADIENT_PRESETS else DEFAULT_GRADIENT_NAME
        self.update()

    def set_error_window(self, low_bound: float, high_bound: float) -> None:
        """Update the current low and high bounds without emitting signals."""
        self._low_bound = max(0.0, min(float(low_bound), 1.0))
        self._high_bound = max(0.0, min(float(high_bound), 1.0))
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the gradient and the current selected error window."""
        super().paintEvent(event)
        painter = QPainter(self)
        bar_rect = self.rect().adjusted(0, 6, 0, -8)
        if bar_rect.width() <= 0 or bar_rect.height() <= 0:
            painter.end()
            return
        for x in range(bar_rect.width()):
            position = x / max(1, bar_rect.width() - 1)
            painter.setPen(interpolate_gradient_color(self._gradient_name, position))
            painter.drawLine(bar_rect.left() + x, bar_rect.top(), bar_rect.left() + x, bar_rect.bottom())

        low_x = self._position_to_x(self._low_bound, bar_rect)
        high_x = self._position_to_x(self._high_bound, bar_rect)
        excluded_color = QColor(0, 0, 0, 110)
        if low_x < high_x:
            painter.fillRect(QRectF(bar_rect.left(), bar_rect.top(), max(0.0, low_x - bar_rect.left()), bar_rect.height()), excluded_color)
            painter.fillRect(QRectF(high_x, bar_rect.top(), max(0.0, bar_rect.right() - high_x), bar_rect.height()), excluded_color)
        elif low_x > high_x:
            painter.fillRect(QRectF(high_x, bar_rect.top(), max(0.0, low_x - high_x), bar_rect.height()), excluded_color)

        painter.setPen(QPen(PANEL_TEXT, 1.0))
        painter.drawRect(bar_rect.adjusted(0, 0, -1, -1))
        self._draw_handle(painter, low_x, bar_rect)
        self._draw_handle(painter, high_x, bar_rect)
        painter.end()

    def mousePressEvent(self, event) -> None:
        """Start dragging the nearest range handle."""
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        bar_rect = self.rect().adjusted(0, 6, 0, -8)
        low_x = self._position_to_x(self._low_bound, bar_rect)
        high_x = self._position_to_x(self._high_bound, bar_rect)
        self._active_handle = 'low' if abs(event.position().x() - low_x) <= abs(event.position().x() - high_x) else 'high'
        self._update_active_handle(event.position().x(), bar_rect)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        """Update the active range handle while dragging."""
        if self._active_handle is None:
            super().mouseMoveEvent(event)
            return
        bar_rect = self.rect().adjusted(0, 6, 0, -8)
        self._update_active_handle(event.position().x(), bar_rect)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        """Finish dragging the active range handle."""
        if event.button() == Qt.MouseButton.LeftButton and self._active_handle is not None:
            self._active_handle = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _draw_handle(self, painter: QPainter, x_pos: float, bar_rect: QRectF) -> None:
        """Draw one vertical handle on the gradient bar."""
        painter.setPen(QPen(PANEL_TEXT, 2.0))
        painter.drawLine(QPointF(x_pos, bar_rect.top() - 3), QPointF(x_pos, bar_rect.bottom() + 3))

    def _position_to_x(self, value: float, bar_rect: QRectF) -> float:
        """Convert a normalized value into a horizontal bar coordinate."""
        return bar_rect.left() + max(0.0, min(float(value), 1.0)) * max(1.0, bar_rect.width() - 1.0)

    def _x_to_position(self, x_pos: float, bar_rect: QRectF) -> float:
        """Convert a horizontal bar coordinate into a normalized value."""
        if bar_rect.width() <= 1.0:
            return 0.0
        return max(0.0, min((x_pos - bar_rect.left()) / (bar_rect.width() - 1.0), 1.0))

    def _update_active_handle(self, x_pos: float, bar_rect: QRectF) -> None:
        """Update the active handle and emit the new selected range."""
        position = self._x_to_position(x_pos, bar_rect)
        if self._active_handle == 'low':
            self._low_bound = position
        elif self._active_handle == 'high':
            self._high_bound = position
        self.update()
        self.rangeEdited.emit(self._low_bound, self._high_bound)
