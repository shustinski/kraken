"""Non-modal Validation profiling diagnostics."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.performance import PerformanceConfig, ProfilingMode
from ..core.profiling import ProfileSnapshot, export_profile


class ProfilingDialog(QDialog):
    configurationChanged = pyqtSignal(object)

    def __init__(self, config: PerformanceConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Validation profiling")
        self.setModal(False)
        self.resize(960, 560)
        self._config = config
        self._snapshot: ProfileSnapshot | None = None

        root = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.mode_combo = QComboBox(self)
        for mode in ProfilingMode:
            self.mode_combo.addItem(mode.value, mode.value)
        self.mode_combo.setCurrentText(config.profiling_mode.value)
        self.mode_combo.currentIndexChanged.connect(self._change_mode)
        controls.addWidget(QLabel("Mode", self))
        controls.addWidget(self.mode_combo)
        controls.addStretch(1)
        export_button = QPushButton("Export snapshot…", self)
        export_button.clicked.connect(self._export_snapshot)
        controls.addWidget(export_button)
        root.addLayout(controls)

        summary = QWidget(self)
        summary_form = QFormLayout(summary)
        self.elapsed_label = QLabel("—", summary)
        self.stage_label = QLabel("—", summary)
        self.frames_label = QLabel("0", summary)
        self.rate_label = QLabel("—", summary)
        self.eta_label = QLabel("—", summary)
        self.cache_label = QLabel("—", summary)
        self.workers_label = QLabel("0", summary)
        self.queue_label = QLabel("0", summary)
        self.ram_label = QLabel("—", summary)
        summary_form.addRow("Elapsed", self.elapsed_label)
        summary_form.addRow("Current stage", self.stage_label)
        summary_form.addRow("Processed frames", self.frames_label)
        summary_form.addRow("Throughput", self.rate_label)
        summary_form.addRow("ETA", self.eta_label)
        summary_form.addRow("Cache hit rate", self.cache_label)
        summary_form.addRow("Workers", self.workers_label)
        summary_form.addRow("Queue", self.queue_label)
        summary_form.addRow("Peak RAM", self.ram_label)
        root.addWidget(summary)

        headers = ("Stage", "Calls", "Total ms", "Self ms", "Avg ms", "Median ms", "P95 ms", "P99 ms", "Share", "Errors")
        self.table = QTableWidget(0, len(headers), self)
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setSortingEnabled(True)
        root.addWidget(self.table, stretch=1)

        self._timer = QTimer(self)
        self._timer.setInterval(config.profiling_ui_refresh_interval_ms)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    @property
    def config(self) -> PerformanceConfig:
        return self._config

    def set_snapshot(self, snapshot: object) -> None:
        if isinstance(snapshot, ProfileSnapshot):
            self._snapshot = snapshot

    def set_config(self, config: PerformanceConfig) -> None:
        self._config = config
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentText(config.profiling_mode.value)
        self.mode_combo.blockSignals(False)
        self._timer.setInterval(config.profiling_ui_refresh_interval_ms)


    def _change_mode(self) -> None:
        mode = ProfilingMode(str(self.mode_combo.currentData() or ProfilingMode.OFF.value))
        self.set_config(replace(self._config, profiling_mode=mode))
        self.configurationChanged.emit(self._config)

    def _refresh(self) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            return
        self.elapsed_label.setText(f"{snapshot.elapsed_ms / 1000.0:.3f} s")
        self.stage_label.setText(snapshot.current_stage or "idle")
        self.frames_label.setText(str(snapshot.processed_frames))
        rate = snapshot.processed_frames / max(0.001, snapshot.elapsed_ms / 1000.0)
        self.rate_label.setText(f"{rate:.2f} frames/s")
        total = int(snapshot.counters.get("frames.total", 0))
        remaining = max(0, total - snapshot.processed_frames)
        self.eta_label.setText(f"{remaining / rate:.1f} s" if total > 0 and rate > 0.0 else "—")
        hits = int(snapshot.counters.get("cache.disk.hits", 0) + snapshot.counters.get("cache.ram.hits", 0))
        misses = int(snapshot.counters.get("cache.disk.misses", 0) + snapshot.counters.get("cache.ram.misses", 0))
        self.cache_label.setText(f"{100.0 * hits / max(1, hits + misses):.1f}%" if hits + misses else "—")
        self.workers_label.setText(str(len(snapshot.workers)))
        queue_depth = int(snapshot.counters.get("worker.queue.depth", snapshot.counters.get("ui.tile.pending", 0)))
        self.queue_label.setText(str(queue_depth))
        peak_ram = snapshot.environment.get("peak_ram_bytes")
        self.ram_label.setText(f"{int(peak_ram) / (1024 * 1024):.1f} MiB" if isinstance(peak_ram, int) else "—")

        sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(snapshot.stages))
        for row_index, row in enumerate(snapshot.stages):
            values = (
                str(row["name"]),
                int(row["calls"]),
                float(row["total_ms"]),
                float(row["self_ms"]),
                float(row["average_ms"] or 0.0),
                float(row["median_ms"] or 0.0),
                float(row["p95_ms"] or 0.0),
                float(row["p99_ms"] or 0.0),
                100.0 * float(row["share"]),
                int(row["errors"]),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value) if column == 0 else f"{value:.3f}" if isinstance(value, float) else str(value))
                if column > 0:
                    item.setData(0, value)
                self.table.setItem(row_index, column, item)
        self.table.setSortingEnabled(sorting)

    def _export_snapshot(self) -> None:
        if self._snapshot is None:
            QMessageBox.information(self, "Validation profiling", "No profiling run is available yet.")
            return
        selected = QFileDialog.getExistingDirectory(self, "Export profiling snapshot", str(self._config.profiling_directory))
        if not selected:
            return
        try:
            exported = export_profile(self._snapshot, Path(selected), self._config)
        except OSError as error:
            QMessageBox.critical(self, "Validation profiling", str(error))
            return
        QMessageBox.information(self, "Validation profiling", f"Exported {len(exported)} files.")


__all__ = ["ProfilingDialog"]
