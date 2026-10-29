"""layer_management: action tab."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Callable
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QPushButton,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kraken_manager.presentation.qt.layer_management_parts.format_scalar import (
    _display_value,
    _local_timestamp,
)
from kraken_manager.presentation.qt.text_bounds import bounded_popup_text, cap_table_row_heights


class ActionTabMixin:
    def _open_temporal_state(self, row: int, _column: int) -> None:
        from kraken_manager.presentation.qt.layer_management import ObjectPropertiesDialog

        loader = self.snapshot.temporal_loader
        if loader is None or not 0 <= row < len(self.snapshot.history):
            return
        historical = loader(self.snapshot.history[row])
        if historical is None:
            return
        ObjectPropertiesDialog(historical, self).exec()

    def _table(self, headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(headers), self)
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.setWordWrap(True)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(headers) - 1, QHeaderView.ResizeMode.Stretch)
        return table

    def _mapping_table(
        self,
        rows: tuple[Mapping[str, object], ...],
        columns: tuple[tuple[str, str], ...],
    ) -> QTableWidget:
        table = self._table(tuple(label for _key, label in columns))
        table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, (key, _label) in enumerate(columns):
                value = values.get(key)
                if key in {"recorded_at", "created_at"} and value:
                    text = _local_timestamp(str(value))
                else:
                    text = _display_value(value)
                item = QTableWidgetItem(text)
                item.setToolTip(bounded_popup_text(text))
                table.setItem(row, column, item)
        table.resizeRowsToContents()
        cap_table_row_heights(table)
        return table

    def _action_tab(
        self,
        table: QTableWidget,
        rows: tuple[Mapping[str, object], ...],
        row_actions: tuple[
            tuple[str, Callable[[Mapping[str, object]], None]], ...
        ],
        *,
        general_actions: tuple[tuple[str, Callable[[], None]], ...] = (),
    ) -> QWidget:
        host = QWidget(self)
        layout = QVBoxLayout(host)
        action_layout = QHBoxLayout()
        for label, callback in general_actions:
            button = QPushButton(label, host)
            button.clicked.connect(lambda _checked=False, action=callback: action())
            action_layout.addWidget(button)
        for label, callback in row_actions:
            button = QPushButton(label, host)

            def invoke(
                _checked: bool = False,
                *,
                action=callback,
                source=table,
                values=rows,
            ) -> None:
                row = source.currentRow()
                if 0 <= row < len(values):
                    action(values[row])

            button.clicked.connect(invoke)
            action_layout.addWidget(button)
        action_layout.addStretch(1)
        layout.addLayout(action_layout)
        layout.addWidget(table, 1)
        return host
