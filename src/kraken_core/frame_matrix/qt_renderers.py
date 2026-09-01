"""Reusable metadata-driven Qt render layers for plugin matrix views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QBrush, QPen

from .models import MatrixItem


class HeatmapLayerRenderer:
    fingerprint = "heatmap-v2"

    def render(self, item: MatrixItem, context) -> None:
        value = item.metadata.get("heatmap_value")
        if not isinstance(value, (int, float)):
            return
        normalized = max(0.0, min(1.0, float(value)))
        explicit_color = QColor(str(item.metadata.get("heatmap_color") or ""))
        if explicit_color.isValid():
            color = explicit_color
            color.setAlphaF(max(0.0, min(1.0, float(item.metadata.get("heatmap_alpha", 0.72)))))
        else:
            # Accessible purple/blue/teal/amber fallback instead of red/green hue rotation.
            stops = (
                (0.0, QColor("#4d2d73")),
                (0.33, QColor("#365c8d")),
                (0.66, QColor("#2d8b8e")),
                (1.0, QColor("#f4be4a")),
            )
            left_position, left_color = stops[0]
            right_position, right_color = stops[-1]
            for index in range(1, len(stops)):
                if normalized <= stops[index][0]:
                    left_position, left_color = stops[index - 1]
                    right_position, right_color = stops[index]
                    break
            ratio = (normalized - left_position) / max(0.000001, right_position - left_position)
            color = QColor(
                int(left_color.red() + (right_color.red() - left_color.red()) * ratio),
                int(left_color.green() + (right_color.green() - left_color.green()) * ratio),
                int(left_color.blue() + (right_color.blue() - left_color.blue()) * ratio),
            )
            color.setAlphaF(0.58)
        context["painter"].fillRect(context["rect"], color)


class StateMarkerLayerRenderer:
    fingerprint = "state-markers-v1"

    def render(self, item: MatrixItem, context) -> None:
        painter, rect = context["painter"], context["rect"]
        if bool(item.metadata.get("excluded")):
            painter.save()
            painter.setPen(QPen(QColor("#94a3b8"), 1.5, Qt.PenStyle.DashLine))
            painter.drawLine(rect.topLeft(), rect.bottomRight())
            painter.drawLine(rect.topRight(), rect.bottomLeft())
            painter.restore()
        if bool(item.metadata.get("reference")):
            painter.save()
            pen = QPen(QColor("#38bdf8"), 3.0)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawRect(rect.adjusted(2.0, 2.0, -2.0, -2.0))
            painter.restore()
        if bool(item.metadata.get("processing")):
            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#facc15")))
            painter.drawEllipse(rect.adjusted(rect.width() - 10.0, 2.0, -2.0, 10.0))
            painter.restore()


class ManagementLayerRenderer:
    fingerprint = "management-v1"

    def render(self, item: MatrixItem, context) -> None:
        color = QColor(str(item.metadata.get("management_color") or ""))
        recommended = bool(item.metadata.get("recommended"))
        if not color.isValid() and not recommended:
            return
        painter, rect = context["painter"], context["rect"]
        painter.save()
        if color.isValid():
            color.setAlpha(80)
            painter.fillRect(rect, color)
        if recommended:
            pen = QPen(QColor("#f8fafc"), 2.5)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawRect(rect.adjusted(1.5, 1.5, -1.5, -1.5))
        painter.restore()


class GridInspectionLayerRenderer:
    fingerprint = "grid-inspection-v1"

    def render(self, item: MatrixItem, context) -> None:
        regions = item.metadata.get("anomaly_regions")
        if not isinstance(regions, Sequence) or isinstance(regions, (str, bytes)):
            return
        painter, rect = context["painter"], context["rect"]
        painter.save()
        pen = QPen(QColor("#ef4444"), 1.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        for region in regions:
            if not isinstance(region, Mapping):
                continue
            try:
                left = rect.left() + float(region["x"]) * rect.width()
                top = rect.top() + float(region["y"]) * rect.height()
                width = float(region["width"]) * rect.width()
                height = float(region["height"]) * rect.height()
            except (KeyError, TypeError, ValueError):
                continue
            painter.drawRect(QRectF(left, top, width, height))
        painter.restore()


class SubcellGridLayerRenderer:
    fingerprint = "subcell-grid-v1"

    def render(self, item: MatrixItem, context) -> None:
        if context.get("semantic_lod") != "subcells":
            return
        values = item.metadata.get("subcells")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
            return
        rows = [row for row in values if isinstance(row, Sequence) and not isinstance(row, (str, bytes))]
        columns = max((len(row) for row in rows), default=0)
        if not rows or columns <= 0:
            return
        painter, rect = context["painter"], context["rect"]
        cell_width = rect.width() / columns
        cell_height = rect.height() / len(rows)
        painter.save()
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                try:
                    normalized = max(0.0, min(1.0, float(value)))
                except (TypeError, ValueError):
                    continue
                color = QColor.fromHsvF((1.0 - normalized) * 0.34, 0.9, 0.95, 0.72)
                painter.fillRect(
                    QRectF(
                        rect.left() + column_index * cell_width,
                        rect.top() + row_index * cell_height,
                        cell_width,
                        cell_height,
                    ),
                    color,
                )
        pen = QPen(QColor(255, 255, 255, 90), 0.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        for column in range(1, columns):
            x = rect.left() + column * cell_width
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
        for row in range(1, len(rows)):
            y = rect.top() + row * cell_height
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
        painter.restore()


DEFAULT_RENDERERS = (
    HeatmapLayerRenderer,
    ManagementLayerRenderer,
    GridInspectionLayerRenderer,
    SubcellGridLayerRenderer,
    StateMarkerLayerRenderer,
)


__all__ = [
    "DEFAULT_RENDERERS",
    "GridInspectionLayerRenderer",
    "HeatmapLayerRenderer",
    "ManagementLayerRenderer",
    "StateMarkerLayerRenderer",
    "SubcellGridLayerRenderer",
]
