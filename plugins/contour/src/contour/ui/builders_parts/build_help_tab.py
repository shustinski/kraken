"""builders: build help tab."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

def build_help_tab(self) -> QWidget:
    tab = QWidget()
    layout = QVBoxLayout(tab)
    self.help_scroll = QScrollArea()
    self.help_scroll.setWidgetResizable(True)
    self.help_container = QWidget()
    self.help_layout = QVBoxLayout(self.help_container)
    self.help_layout.setContentsMargins(0, 0, 0, 0)
    self.help_scroll.setWidget(self.help_container)
    layout.addWidget(self.help_scroll, 1)
    self._rebuild_help_cards()
    return tab
