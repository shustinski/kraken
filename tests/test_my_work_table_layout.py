"""My Work columns keep headers and action buttons inside the visible row."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from kraken_hub.presentation.qt.parts.my_work_panel import _fit_my_work_table
from kraken_manager.presentation.qt.text_bounds import TABLE_ROW_HEIGHT_LIMIT


def test_headers_and_buttons_get_a_bounded_row(qapp) -> None:
    from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QWidget

    del qapp
    headers = ("Тип", "Проект", "Срок / обновление", "Действия")
    table = QTableWidget(1, len(headers))
    table.setHorizontalHeaderLabels(headers)
    for column, label in enumerate(headers[:-1]):
        table.setItem(column and 0, column, QTableWidgetItem(label * 8))
    host = QWidget()
    layout = QHBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    for label in ("Повторно выгрузить", "На доработку", "Отменить"):
        layout.addWidget(QPushButton(label, host))
    table.setCellWidget(0, 3, host)
    table.setRowHeight(0, 36974)
    _fit_my_work_table(table)
    header = table.horizontalHeader()
    for column, label in enumerate(headers):
        assert table.columnWidth(column) >= header.sectionSizeHint(column)
        assert label
    assert table.rowHeight(0) <= TABLE_ROW_HEIGHT_LIMIT
    assert table.columnWidth(3) >= host.sizeHint().width()
