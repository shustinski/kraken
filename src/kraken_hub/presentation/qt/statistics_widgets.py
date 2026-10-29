"""Small dependency-free Qt widgets for the statistics dashboard."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QPaintEvent, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QToolTip, QWidget

from kraken_manager.infrastructure.reports import (
    MetricChartKind,
    MetricDefinition,
    ReportBucket,
    format_metric_value,
)


class MetricChartWidget(QWidget):
    """Compact themed time-series card rendered with stock Qt."""

    def __init__(self, definition: MetricDefinition, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.definition = definition
        self._points: tuple[tuple[str, float, str], ...] = ()
        self._total = format_metric_value(definition, 0)
        self._plot_rect = QRectF()
        self.setObjectName(f"statisticsChart_{definition.key}")
        self.setAccessibleName(definition.label)
        self.setToolTip(definition.description)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 210)

    @property
    def point_count(self) -> int:
        return len(self._points)

    @property
    def point_labels(self) -> tuple[str, ...]:
        return tuple(point[0] for point in self._points)

    @property
    def total_text(self) -> str:
        return self._total

    def set_series(self, buckets: tuple[ReportBucket, ...], total: int | float) -> None:
        self._points = tuple(
            (
                bucket.label,
                float(bucket.metrics.values.get(self.definition.key, 0)),
                format_metric_value(
                    self.definition,
                    bucket.metrics.values.get(self.definition.key, 0),
                ),
            )
            for bucket in buckets
        )
        self._total = format_metric_value(self.definition, total)
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        card = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        painter.setPen(QPen(palette.color(palette.ColorRole.Mid), 1.0))
        painter.setBrush(palette.color(palette.ColorRole.Base))
        painter.drawRoundedRect(card, 9.0, 9.0)

        title_rect = card.adjusted(14.0, 10.0, -14.0, -card.height() + 34.0)
        title_font = painter.font()
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(palette.color(palette.ColorRole.Text))
        title = painter.fontMetrics().elidedText(
            self.definition.label,
            Qt.TextElideMode.ElideRight,
            int(title_rect.width() * 0.7),
        )
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, self._total)

        self._plot_rect = card.adjusted(48.0, 46.0, -16.0, -32.0)
        raw_values = tuple(point[1] for point in self._points)
        if not raw_values or not any(value != 0.0 for value in raw_values):
            painter.setFont(self.font())
            painter.setPen(palette.color(palette.ColorRole.PlaceholderText))
            painter.drawText(
                self._plot_rect,
                Qt.AlignmentFlag.AlignCenter,
                "Нет данных за выбранный период",
            )
            return

        maximum = max(raw_values)
        if maximum <= 0:
            maximum = 1.0
        painter.setFont(self.font())
        grid_color = QColor(palette.color(palette.ColorRole.Mid))
        grid_color.setAlpha(95)
        painter.setPen(QPen(grid_color, 1.0))
        for step in range(4):
            y = self._plot_rect.bottom() - self._plot_rect.height() * step / 3.0
            painter.drawLine(
                QPointF(self._plot_rect.left(), y),
                QPointF(self._plot_rect.right(), y),
            )

        label_color = palette.color(palette.ColorRole.PlaceholderText)
        painter.setPen(label_color)
        maximum_label = format_metric_value(self.definition, maximum)
        painter.drawText(
            QRectF(card.left() + 4.0, self._plot_rect.top() - 8.0, 40.0, 18.0),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            maximum_label,
        )
        painter.drawText(
            QRectF(card.left() + 4.0, self._plot_rect.bottom() - 9.0, 40.0, 18.0),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            format_metric_value(self.definition, 0),
        )

        accent = QColor("#3B82F6")
        point_positions = self._point_positions(raw_values, maximum)
        if self.definition.chart_kind is MetricChartKind.BAR:
            slot_width = self._plot_rect.width() / max(1, len(point_positions))
            bar_width = max(2.0, min(22.0, slot_width * 0.66))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(accent)
            for position in point_positions:
                painter.drawRoundedRect(
                    QRectF(
                        position.x() - bar_width / 2.0,
                        position.y(),
                        bar_width,
                        self._plot_rect.bottom() - position.y(),
                    ),
                    min(3.0, bar_width / 2.0),
                    min(3.0, bar_width / 2.0),
                )
        else:
            path = QPainterPath()
            for index, position in enumerate(point_positions):
                if index == 0:
                    path.moveTo(position)
                else:
                    path.lineTo(position)
            painter.setPen(QPen(accent, 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            painter.setBrush(accent)
            painter.setPen(Qt.PenStyle.NoPen)
            for position in point_positions:
                painter.drawEllipse(position, 3.0, 3.0)

        self._draw_period_labels(painter, label_color, point_positions)

    def _point_positions(self, values: tuple[float, ...], maximum: float) -> tuple[QPointF, ...]:
        x_values: tuple[float, ...]
        if len(values) == 1:
            x_values = (self._plot_rect.center().x(),)
        else:
            x_values = tuple(
                self._plot_rect.left() + self._plot_rect.width() * index / (len(values) - 1)
                for index in range(len(values))
            )
        return tuple(
            QPointF(
                x,
                self._plot_rect.bottom() - self._plot_rect.height() * max(0.0, value) / maximum,
            )
            for x, value in zip(x_values, values, strict=True)
        )

    def _draw_period_labels(
        self,
        painter: QPainter,
        color: QColor,
        positions: tuple[QPointF, ...],
    ) -> None:
        if not positions:
            return
        indices = sorted({0, len(positions) // 2, len(positions) - 1})
        painter.setPen(color)
        metrics = painter.fontMetrics()
        for index in indices:
            label = metrics.elidedText(
                self._points[index][0],
                Qt.TextElideMode.ElideRight,
                90,
            )
            label_rect = QRectF(
                positions[index].x() - 47.0,
                self._plot_rect.bottom() + 5.0,
                94.0,
                20.0,
            )
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if event is None:
            return
        if not self._points or not self._plot_rect.contains(event.position()):
            QToolTip.hideText()
            super().mouseMoveEvent(event)
            return
        if len(self._points) == 1:
            index = 0
        else:
            relative = (event.position().x() - self._plot_rect.left()) / self._plot_rect.width()
            index = min(len(self._points) - 1, max(0, round(relative * (len(self._points) - 1))))
        label, _value, formatted = self._points[index]
        QToolTip.showText(event.globalPosition().toPoint(), f"{label}\n{formatted}", self)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QEvent | None) -> None:  # noqa: N802
        QToolTip.hideText()
        super().leaveEvent(event)


__all__ = ["MetricChartWidget"]
