"""Persistent standalone analysis history viewer."""

from __future__ import annotations

import json

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..storage import AnalysisHistoryStore


class StandaloneHistoryDialog(QDialog):
    repeatRequested = pyqtSignal(object)

    def __init__(self, store: AnalysisHistoryStore | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store or AnalysisHistoryStore()
        self.setWindowTitle("История анализа Karakal")
        self.resize(900, 500)
        root = QVBoxLayout(self)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(("Запуск", "Статус", "Кадры", "Готово", "Ошибки", "Fingerprint"))
        root.addWidget(self.table, 1)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(
            self.details.textInteractionFlags() | Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self.details)
        actions = QHBoxLayout()
        self.repeat_button = QPushButton("Повторить запуск")
        actions.addWidget(self.repeat_button)
        actions.addStretch(1)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        actions.addWidget(close)
        root.addLayout(actions)
        close.rejected.connect(self.reject)
        self.table.currentCellChanged.connect(lambda *_: self._show_selected())
        self.repeat_button.clicked.connect(self._repeat_selected)
        self.reload()

    def reload(self) -> None:
        runs = self.store.list_runs()
        self.table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            values = (
                run.run_id,
                run.state,
                run.total_frames,
                run.completed_frames,
                run.failed_frames,
                run.fingerprint,
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
        if runs:
            self.table.selectRow(0)
        self._show_selected()

    def _selected_run(self):
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        return None if item is None else self.store.get_run(item.text())

    def _show_selected(self) -> None:
        run = self._selected_run()
        if run is None:
            self.details.clear()
            self.repeat_button.setEnabled(False)
            return
        sources = ", ".join(
            f"{source.binding_key}={source.display_name or source.source_id}@{source.source_version}"
            for source in run.manifest.source_bindings
        )
        recipe = json.dumps(run.manifest.recipe.to_payload(), ensure_ascii=False, sort_keys=True)
        self.details.setText(f"Модели: {sources}\nРецепт: {recipe}")
        self.repeat_button.setEnabled(True)

    def _repeat_selected(self) -> None:
        run = self._selected_run()
        if run is not None:
            self.repeatRequested.emit(run.manifest)


__all__ = ["StandaloneHistoryDialog"]
