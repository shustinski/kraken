"""Minimal diagnostic UI: load a grid, run a 3×3 slice, show mosaic and pair scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from kraken_core.theme import add_theme_menu

from cartograph.application.nominal import PlacementSettings
from cartograph.application.pipeline import RunLocalVerticalSlice, VerticalSliceRequest
from cartograph.domain.coordinates import GridCoordinate, NominalPlacementMode
from cartograph.domain.errors import CartographError
from cartograph.infrastructure.render import BlendMode


class CartographWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cartograph")
        self._mosaic_array: np.ndarray | None = None
        self._pipeline = RunLocalVerticalSlice()
        add_theme_menu(self)

        root = QWidget(self)
        layout = QVBoxLayout(root)

        form = QFormLayout()
        self._path = QLineEdit()
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self._path)
        path_row.addWidget(browse)
        path_widget = QWidget()
        path_widget.setLayout(path_row)
        form.addRow("Grid", path_widget)

        self._center_row = QSpinBox()
        self._center_row.setMaximum(10_000)
        self._center_col = QSpinBox()
        self._center_col.setMaximum(10_000)
        form.addRow("Center row", self._center_row)
        form.addRow("Center col", self._center_col)

        self._overlap_x = QDoubleSpinBox()
        self._overlap_x.setRange(0.0, 0.9)
        self._overlap_x.setSingleStep(0.05)
        self._overlap_x.setValue(0.1)
        self._overlap_y = QDoubleSpinBox()
        self._overlap_y.setRange(0.0, 0.9)
        self._overlap_y.setSingleStep(0.05)
        self._overlap_y.setValue(0.1)
        form.addRow("Overlap X", self._overlap_x)
        form.addRow("Overlap Y", self._overlap_y)

        self._placement = QComboBox()
        for mode in NominalPlacementMode:
            self._placement.addItem(mode.value)
        form.addRow("Placement", self._placement)

        self._blend = QComboBox()
        for mode in BlendMode:
            self._blend.addItem(mode.value)
        form.addRow("Blend", self._blend)

        layout.addLayout(form)
        run = QPushButton("Register 3×3")
        run.clicked.connect(self._run)
        layout.addWidget(run)

        self._status = QLabel("Load a tile folder or grid.json, then register a 3×3 window.")
        layout.addWidget(self._status)
        self._mosaic = QLabel()
        self._mosaic.setMinimumSize(320, 240)
        layout.addWidget(self._mosaic)

        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            ["pair", "dx", "dy", "confidence", "zncc", "grad_zncc", "cycle", "status"]
        )
        layout.addWidget(self._table)
        self.setCentralWidget(root)
        self.resize(960, 800)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select SEM tile folder")
        if folder:
            self._path.setText(folder)

    def _run(self) -> None:
        path_text = self._path.text().strip()
        if not path_text:
            QMessageBox.warning(self, "Cartograph", "Select a tile folder or grid.json first.")
            return
        request = VerticalSliceRequest(
            path=Path(path_text),
            center=GridCoordinate(self._center_row.value(), self._center_col.value()),
            overlap_x=self._overlap_x.value(),
            overlap_y=self._overlap_y.value(),
            placement=PlacementSettings(mode=NominalPlacementMode(self._placement.currentText())),
            blend=BlendMode(self._blend.currentText()),
        )
        try:
            result = self._pipeline.execute(request)
        except (CartographError, ValueError, OSError) as exc:
            QMessageBox.critical(self, "Cartograph", str(exc))
            return
        self._status.setText(
            f"status={result.solution.status.value} cache={result.outcome.from_cache} "
            f"edges={len(result.solution.graph.edges)}"
        )
        self._show_mosaic(result.mosaic.pixels)
        self._fill_table(result.solution.graph.edges)

    def _show_mosaic(self, pixels: np.ndarray) -> None:
        array = np.ascontiguousarray(np.clip(pixels, 0, 255).astype(np.uint8))
        self._mosaic_array = array
        qimage = QImage(
            array.data,
            int(array.shape[1]),
            int(array.shape[0]),
            int(array.strides[0]),
            QImage.Format.Format_Grayscale8,
        )
        self._mosaic.setPixmap(QPixmap.fromImage(qimage).scaled(720, 540))

    def _fill_table(self, edges) -> None:
        self._table.setRowCount(len(edges))
        for row, edge in enumerate(edges):
            result = edge.result
            pair = f"{edge.source.row},{edge.source.col} → {edge.target.row},{edge.target.col}"
            values = [
                pair,
                f"{result.transform.dx:.3f}",
                f"{result.transform.dy:.3f}",
                f"{result.confidence:.3f}",
                f"{result.raw_zncc:.3f}",
                f"{result.gradient_zncc:.3f}",
                "" if result.cycle_residual is None else f"{result.cycle_residual:.3f}",
                result.status.value,
            ]
            for column, value in enumerate(values):
                self._table.setItem(row, column, QTableWidgetItem(value))
        self._table.resizeColumnsToContents()
