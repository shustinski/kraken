from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from kategb.presentation.qt.window import KateGBWindow


def test_kategb_window_uses_russian_ui_text() -> None:
    app = QApplication.instance() or QApplication([])
    window = KateGBWindow()
    try:
        assert window.tabs.tabText(0) == "Выборка"
        assert window.tabs.tabText(1) == "Проверка"
        assert window.load_markup_button.text() == "Загрузить разметку"
        assert window.generate_button.text() == "Сгенерировать кадры"
        assert window.results_table.horizontalHeaderItem(0).text() == "Исполнитель"
        assert window.status.text() == "Готово."
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
