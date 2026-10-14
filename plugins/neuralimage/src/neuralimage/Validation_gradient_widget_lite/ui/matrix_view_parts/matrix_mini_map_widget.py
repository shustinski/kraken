"""matrix_view: matrix mini map widget."""
from __future__ import annotations
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QImage, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import (
    QWidget,
)
from neuralimage.Validation_gradient_widget_lite.ui.ui_constants import (
    MINIMAP_FRAME_MARGIN,
    MINIMAP_MIN_SIZE,
    MINIMAP_PROCESSING_TRIANGLE_HALF_WIDTH,
    MINIMAP_PROCESSING_TRIANGLE_HEIGHT,
    MINIMAP_REFERENCE_MARKER_SIDE,
    MINIMAP_REFERENCE_PEN_WIDTH,
    MINIMAP_SELECTED_COLOR,
    MINIMAP_SELECTED_OUTLINE_WIDTH,
    MINIMAP_SELECTED_RADIUS_OFF,
    MINIMAP_SELECTED_RADIUS_ON,
    PANEL_BACKGROUND,
    PANEL_TEXT,
    PROCESSING_FILL,
    REFERENCE_BORDER,
    VISIBLE_RECT_MIN_SIZE,
)

class MatrixMiniMapWidget(QWidget):
    """Render a compact overview image for the active matrix tab."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._visible_rect = QRectF()
        self._selected_position: tuple[int, int] | None = None
        self._selected_blink_on = False
        self._processing_positions: tuple[tuple[int, int], ...] = ()
        self._reference_position: tuple[int, int] | None = None
        self.setMinimumSize(*MINIMAP_MIN_SIZE)

    def set_overview(self, image, visible_rect, selected_position, selected_blink_on, processing_positions, reference_position) -> None:
        self._image = image
        self._visible_rect = QRectF(visible_rect)
        self._selected_position = selected_position
        self._selected_blink_on = bool(selected_blink_on)
        self._processing_positions = tuple(processing_positions)
        self._reference_position = reference_position
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), PANEL_BACKGROUND)
        if self._image is None or self._image.isNull():
            painter.setPen(PANEL_TEXT)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._t("matrix.no_matrix"))
            painter.end()
            return
        target = self.rect().adjusted(MINIMAP_FRAME_MARGIN, MINIMAP_FRAME_MARGIN, -MINIMAP_FRAME_MARGIN, -MINIMAP_FRAME_MARGIN)
        painter.drawImage(target, self._image)
        width = max(1, self._image.width())
        height = max(1, self._image.height())
        cell_w = target.width() / width
        cell_h = target.height() / height
        if not self._visible_rect.isNull():
            rect = QRectF(
                target.left() + self._visible_rect.left() * target.width(),
                target.top() + self._visible_rect.top() * target.height(),
                max(VISIBLE_RECT_MIN_SIZE, self._visible_rect.width() * target.width()),
                max(VISIBLE_RECT_MIN_SIZE, self._visible_rect.height() * target.height()),
            )
            painter.setPen(QPen(PANEL_TEXT, 1.5))
            painter.drawRect(rect)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(PROCESSING_FILL)
        for row, column in self._processing_positions:
            painter.drawEllipse(
                QPointF(target.left() + (column + 0.5) * cell_w, target.top() + (row + 0.5) * cell_h),
                MINIMAP_PROCESSING_TRIANGLE_HALF_WIDTH,
                MINIMAP_PROCESSING_TRIANGLE_HEIGHT,
            )
        if self._reference_position is not None:
            row, column = self._reference_position
            painter.setPen(QPen(REFERENCE_BORDER, MINIMAP_REFERENCE_PEN_WIDTH))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(
                QRectF(
                    target.left() + column * cell_w,
                    target.top() + row * cell_h,
                    max(MINIMAP_REFERENCE_MARKER_SIDE, cell_w),
                    max(MINIMAP_REFERENCE_MARKER_SIDE, cell_h),
                )
            )
        if self._selected_position is not None:
            row, column = self._selected_position
            radius = MINIMAP_SELECTED_RADIUS_ON if self._selected_blink_on else MINIMAP_SELECTED_RADIUS_OFF
            painter.setPen(QPen(MINIMAP_SELECTED_COLOR, MINIMAP_SELECTED_OUTLINE_WIDTH))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                QPointF(target.left() + (column + 0.5) * cell_w, target.top() + (row + 0.5) * cell_h),
                radius,
                radius,
            )
        painter.end()
