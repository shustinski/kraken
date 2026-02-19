from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel


class TrainingMetricsDock(QDockWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__('Training Metrics', parent)
        self.setObjectName('trainingMetricsDock')
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._train_epoch_points: list[tuple[float, float]] = []
        self._val_epoch_points: list[tuple[float, float]] = []
        self._batch_points: list[tuple[float, float]] = []

        self._pg = None
        self._epoch_plot = None
        self._batch_plot = None
        self._train_curve = None
        self._val_curve = None
        self._batch_curve = None

        try:
            import pyqtgraph as pg

            self._pg = pg
            pg.setConfigOptions(antialias=True)

            self._epoch_plot = pg.PlotWidget(title='Loss vs Epoch')
            self._batch_plot = pg.PlotWidget(title='Train Loss vs Batch (Current Epoch)')

            for plot in (self._epoch_plot, self._batch_plot):
                plot.showGrid(x=True, y=True, alpha=0.3)
                plot.setBackground('#10161d')
                plot.getAxis('left').setTextPen('#c9d7e8')
                plot.getAxis('bottom').setTextPen('#c9d7e8')

            if self._epoch_plot is not None:
                self._epoch_plot.addLegend()
            self._train_curve = self._epoch_plot.plot(
                pen=pg.mkPen('#50baff', width=2),
                name='Train Loss',
            )
            self._val_curve = self._epoch_plot.plot(
                pen=pg.mkPen('#ffaa5c', width=2),
                name='Val Loss',
            )
            self._batch_curve = self._batch_plot.plot(pen=pg.mkPen('#89e47d', width=2))

            layout.addWidget(self._epoch_plot)
            layout.addWidget(self._batch_plot)
        except Exception:
            layout.addWidget(QLabel('pyqtgraph is not available. Install pyqtgraph to enable live charts.'))

        self.setWidget(container)

    def clear(self):
        self._train_epoch_points.clear()
        self._val_epoch_points.clear()
        self._batch_points.clear()
        if self._train_curve is not None:
            self._train_curve.setData([], [])
        if self._val_curve is not None:
            self._val_curve.setData([], [])
        if self._batch_curve is not None:
            self._batch_curve.setData([], [])
        if self._batch_plot is not None:
            self._batch_plot.setTitle('Train Loss vs Batch (Current Epoch)')

    def add_train_epoch_point(self, epoch: int, loss: float):
        self._train_epoch_points.append((float(epoch), float(loss)))
        if self._train_curve is not None:
            xs = [p[0] for p in self._train_epoch_points]
            ys = [p[1] for p in self._train_epoch_points]
            self._train_curve.setData(xs, ys)

    def add_val_epoch_point(self, epoch: int, loss: float):
        self._val_epoch_points.append((float(epoch), float(loss)))
        if self._val_curve is not None:
            xs = [p[0] for p in self._val_epoch_points]
            ys = [p[1] for p in self._val_epoch_points]
            self._val_curve.setData(xs, ys)

    def set_batch_points(self, epoch: int, points: list[tuple[float, float]]):
        self._batch_points = list(points)
        if self._batch_curve is not None:
            xs = [p[0] for p in self._batch_points]
            ys = [p[1] for p in self._batch_points]
            self._batch_curve.setData(xs, ys)
        if self._batch_plot is not None:
            self._batch_plot.setTitle(f'Train Loss vs Batch (Epoch {epoch})')


# Backward-compatible alias for existing imports.
TrainingMetricsPanel = TrainingMetricsDock
