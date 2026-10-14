"""details_dialog: overlay graphics view."""
from __future__ import annotations
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsScene,
    QGraphicsView,
)

class _OverlayGraphicsView(QGraphicsView):
    """Provide zoom and middle-button preview toggling for the overlay preview."""

    middlePressed = pyqtSignal()
    middleReleased = pyqtSignal()
    leftClicked = pyqtSignal(QPointF)
    viewTransformChanged = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(92, 96, 102))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setMinimumSize(320, 320)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        current_scale = self.transform().m11()
        next_scale = current_scale * factor
        if 0.05 <= next_scale <= 60.0:
            self.scale(factor, factor)
            self.viewTransformChanged.emit()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.middlePressed.emit()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.leftClicked.emit(self.mapToScene(event.position().toPoint()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.middleReleased.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def fit_to_scene(self) -> None:
        scene = self.scene()
        if scene is None:
            return
        fit_rect = scene.itemsBoundingRect()
        if fit_rect.isNull():
            fit_rect = scene.sceneRect()
        if fit_rect.isNull() or self.viewport().width() <= 4 or self.viewport().height() <= 4:
            return
        fit_rect = fit_rect.adjusted(-2.0, -2.0, 2.0, 2.0)
        self.fitInView(fit_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(fit_rect.center())

    def reset_view(self) -> None:
        self.resetTransform()
        self.fit_to_scene()
        self.viewTransformChanged.emit()
