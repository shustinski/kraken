"""main_window: construction."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget,
)


class ConstructionMixin:
    def __init__(self, row: int, item: dict, texts: dict[str, str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._row = int(row)
        self._texts = texts
        self._setup_ui()
        self.update_item(row, item, texts)
