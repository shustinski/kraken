"""Legend for performer, time, quality and status on the frame thumbnails."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from kraken_manager.domain.layer_board import LegendSection, matrix_legend


class MatrixLegendPanel(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("matrixLegend")
        self.setMinimumWidth(210)
        self.setMaximumWidth(280)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(10)
        self._sections = None
        self.set_rows(())

    def set_rows(self, rows: Sequence[dict[str, object]]) -> None:
        sections = matrix_legend(tuple(rows))
        if sections == self._sections:
            return
        self._sections = sections
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        for section in sections:
            widget = _section_widget(section, self)
            self._layout.addWidget(widget)
            if self.isVisible():
                widget.show()
        self._layout.addStretch(1)
        self._layout.activate()


def _section_widget(section: LegendSection, parent: QWidget) -> QWidget:
    host = QWidget(parent)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    title = QLabel(section.title, host)
    title.setObjectName("matrixLegendTitle")
    layout.addWidget(title)
    if section.empty:
        empty = QLabel(section.empty_text, host)
        empty.setObjectName("matrixLegendEmpty")
        empty.setStyleSheet("color:#94a3b8;")
        layout.addWidget(empty)
        return host
    for entry in section.entries:
        row = QHBoxLayout()
        swatch = QLabel(host)
        swatch.setFixedSize(14, 14)
        color = QColor(entry.color) if entry.color else QColor("#334155")
        if not color.isValid():
            color = QColor("#334155")
        swatch.setStyleSheet(f"background:{color.name()}; border:1px solid #0f172a;")
        if not entry.color:
            swatch.setToolTip("Цвет не задан")
        label = QLabel(entry.label, host)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(swatch)
        row.addWidget(label, 1)
        layout.addLayout(row)
    return host
