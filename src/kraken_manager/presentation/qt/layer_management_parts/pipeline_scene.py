"""layer_management: pipeline scene."""
from __future__ import annotations
from PyQt6.QtCore import QLineF, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QGraphicsScene,
)

class PipelineScene(QGraphicsScene):
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # type: ignore[override]
        from kraken_core.theme import scene_color

        painter.fillRect(rect, QColor(scene_color("pipeline")))
        minor, major = 20, 100
        for spacing, color in ((minor, scene_color("grid_minor")), (major, scene_color("grid_major"))):
            pen = QPen(QColor(color), 0)
            painter.setPen(pen)
            left = int(rect.left()) - (int(rect.left()) % spacing)
            top = int(rect.top()) - (int(rect.top()) % spacing)
            for x in range(left, int(rect.right()) + spacing, spacing):
                painter.drawLine(QLineF(float(x), rect.top(), float(x), rect.bottom()))
            for y in range(top, int(rect.bottom()) + spacing, spacing):
                painter.drawLine(QLineF(rect.left(), float(y), rect.right(), float(y)))
