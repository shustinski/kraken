"""matrix_view: gradient preview bar."""
from __future__ import annotations
from PyQt6.QtGui import QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import (
    QWidget,
)
from neuralimage.Validation_gradient_widget_lite.ui.ui_constants import (
    DEFAULT_GRADIENT_NAME,
    GRADIENT_PRESETS,
    GRADIENT_PREVIEW_MIN_HEIGHT,
    SUBDUED_TEXT_COLOR,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.compute_auto_color_window import interpolate_gradient_color

class _GradientPreviewBar(QWidget):
    """Render a compact horizontal preview for one gradient preset."""

    def __init__(self, parent=None) -> None:
        """Initialize the preview bar."""
        super().__init__(parent)
        self._gradient_name = DEFAULT_GRADIENT_NAME
        self.setMinimumHeight(GRADIENT_PREVIEW_MIN_HEIGHT)

    def set_gradient_name(self, gradient_name: str) -> None:
        """Set the gradient shown in the preview bar."""
        self._gradient_name = gradient_name if gradient_name in GRADIENT_PRESETS else DEFAULT_GRADIENT_NAME
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the current gradient preview."""
        super().paintEvent(event)
        painter = QPainter(self)
        target = self.rect().adjusted(0, 2, 0, -2)
        if target.width() <= 0 or target.height() <= 0:
            painter.end()
            return
        for x in range(target.width()):
            position = x / max(1, target.width() - 1)
            painter.setPen(interpolate_gradient_color(self._gradient_name, position))
            painter.drawLine(target.left() + x, target.top(), target.left() + x, target.bottom())
        painter.setPen(QPen(SUBDUED_TEXT_COLOR, 1.0))
        painter.drawRect(target.adjusted(0, 0, -1, -1))
        painter.end()
