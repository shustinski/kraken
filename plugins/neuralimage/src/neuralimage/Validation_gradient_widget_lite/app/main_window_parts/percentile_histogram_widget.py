"""main_window: percentile histogram widget."""
from __future__ import annotations
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)
from ...ui.i18n import Translator
from ...ui.ui_constants import (
    PERCENTILE_BAND_BOUNDS,
    PERCENTILE_BAND_COLORS,
    PERCENTILE_BAND_LABELS,
    PERCENTILE_BAND_TITLES,
)

class _PercentileHistogramWidget(QWidget):
    """Draw a compact histogram over fixed percentile bins."""

    binClicked = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._counts = [0] * len(PERCENTILE_BAND_BOUNDS)
        self._total = 0
        self._active_bin: int | None = None
        self.setMinimumHeight(120)

    def set_payload(self, counts: list[int], total: int, *, active_bin: int | None = None) -> None:
        expected = len(PERCENTILE_BAND_BOUNDS)
        self._counts = [int(value) for value in counts[:expected]] + [0] * max(0, expected - len(counts))
        self._counts = self._counts[:expected]
        self._total = int(total)
        self._active_bin = None if active_bin is None else int(active_bin)
        self.update()

    def _chart_rect(self) -> QRectF:
        return QRectF(self.rect().adjusted(6, 8, -6, -20))

    def _bar_rects(self, rect: QRectF) -> list[QRectF]:
        if rect.width() <= 0 or rect.height() <= 0:
            return []
        max_count = max(1, max(self._counts, default=1))
        band_count = max(1, len(self._counts))
        bar_width = max(12.0, rect.width() / max(4.0, band_count * 1.25))
        gap = max(6.0, (rect.width() - bar_width * band_count) / max(1.0, band_count - 1.0))
        rects: list[QRectF] = []
        for index, count in enumerate(self._counts):
            height_ratio = float(count) / float(max_count)
            bar_height = max(2.0, (rect.height() - 24.0) * height_ratio) if count > 0 else 2.0
            left = rect.left() + index * (bar_width + gap)
            top = rect.bottom() - 18.0 - bar_height
            rects.append(QRectF(left, top, bar_width, bar_height))
        return rects

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self._chart_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            painter.end()
            return
        painter.fillRect(rect, QColor('#11161d'))
        labels = PERCENTILE_BAND_LABELS
        for index, bar_rect in enumerate(self._bar_rects(rect)):
            count = self._counts[index]
            color = QColor(PERCENTILE_BAND_COLORS[index])
            painter.fillRect(bar_rect, color)
            if self._active_bin == index:
                painter.setPen(QPen(QColor('#f5f8fc'), 2.0))
                painter.drawRect(bar_rect.adjusted(0.5, 0.5, -0.5, -0.5))
            painter.setPen(QPen(QColor('#dce7f3')))
            painter.drawText(QRectF(bar_rect.left() - 6.0, bar_rect.top() - 16.0, bar_rect.width() + 12.0, 14.0), Qt.AlignmentFlag.AlignCenter, str(count))
            painter.drawText(QRectF(bar_rect.left() - 8.0, rect.bottom() - 16.0, bar_rect.width() + 16.0, 14.0), Qt.AlignmentFlag.AlignCenter, labels[index])
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        point = event.position()
        for index, bar_rect in enumerate(self._bar_rects(self._chart_rect())):
            if bar_rect.contains(point):
                self.binClicked.emit(index)
                event.accept()
                return
        super().mousePressEvent(event)

class _PercentileHistogramCard(QWidget):
    """Show one metric percentile distribution as a compact chart card."""

    binClicked = pyqtSignal(str, int)

    def __init__(self, title: str, metric_key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._metric_key = str(metric_key)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.setStyleSheet('background-color: #1a2028; border: 1px solid #304050; border-radius: 10px;')
        self.title_label = QLabel(title, self)
        self.title_label.setWordWrap(True)
        self.summary_label = QLabel('-', self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet('color: #c9d3df; background: transparent; border: none;')
        self.chart = _PercentileHistogramWidget(self)
        self.chart.binClicked.connect(lambda index: self.binClicked.emit(self._metric_key, index))
        layout.addWidget(self.title_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.chart)

    def set_payload(self, title: str, counts: list[int], total: int, *, visible: bool, active_bin: int | None = None, tooltip: str = '') -> None:
        self.setVisible(visible)
        self.title_label.setText(title)
        self.chart.set_payload(counts, total, active_bin=active_bin)
        labels = PERCENTILE_BAND_TITLES
        translator = Translator()
        summary = ' | '.join(f'{label}: {int(count)}' for label, count in zip(labels, counts))
        if active_bin is not None and 0 <= int(active_bin) < len(labels):
            summary = f'{translator.tr("hist.filter_active")}: {labels[int(active_bin)]} | ' + summary
        self.summary_label.setText(summary)
        self.setToolTip(tooltip)
        self.title_label.setToolTip(tooltip)
        self.summary_label.setToolTip(tooltip)
        self.chart.setToolTip(tooltip)
