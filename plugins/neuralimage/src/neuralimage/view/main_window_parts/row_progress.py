"""main_window: row progress."""
from __future__ import annotations
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import (
    QProgressBar,
)


class RowProgressMixin:
    @staticmethod
    def _set_row_progress(progress_bar: QProgressBar, title: str, current: int, total: int) -> None:
        bounded_current = max(0, int(current))
        bounded_total = max(0, int(total))
        value = max(0, min(100, int((bounded_current / bounded_total) * 100))) if bounded_total else 0
        progress_bar.setValue(value)
        text = f'{title}: {value}% ({bounded_current}/{bounded_total})' if bounded_total else f'{title}: 0%'
        progress_bar.setFormat(text)
        text_width = QFontMetrics(progress_bar.font()).horizontalAdvance(text)
        progress_bar.setFixedWidth(max(120, text_width + 28))

    def _fit_name_edit_to_text(self) -> None:
        text = self.name_edit.text() or self.name_edit.placeholderText() or 'Task'
        text_width = QFontMetrics(self.name_edit.font()).horizontalAdvance(text)
        self.name_edit.setFixedWidth(max(80, text_width + 28))
