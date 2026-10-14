"""details_dialog: expandable detail score card."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

class _ExpandableDetailScoreCard(QWidget):
    """Show the current frame score with click-to-expand details."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.title_label = QLabel(title, self)
        self.title_label.setWordWrap(True)
        self.value_button = QPushButton("-", self)
        self.value_button.setCheckable(True)
        self.value_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.value_button.setMinimumHeight(40)
        self.value_button.setStyleSheet(
            "padding: 8px 12px; border-radius: 10px; "
            "background-color: #2f3844; color: #edf3fb; font-weight: 700; "
            "border: none; text-align: center;"
        )
        self.details_label = QLabel("", self)
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet(
            "padding: 6px 8px; color: #c9d3df; "
            "background-color: #11161d; border-radius: 8px;"
        )
        self.details_label.hide()
        self.value_button.toggled.connect(self._on_toggled)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_button)
        layout.addWidget(self.details_label)

    def _on_toggled(self, checked: bool) -> None:
        self.details_label.setVisible(bool(checked) and bool(self.details_label.text().strip()))

    def set_payload(self, title: str, value_text: str, value_style: str, details: str, tooltip: str = "") -> None:
        self.title_label.setText(str(title))
        self.value_button.setText(str(value_text))
        self.value_button.setStyleSheet(str(value_style) + "; border: none; text-align: center;")
        self.details_label.setText(str(details))
        self.details_label.setVisible(bool(self.value_button.isChecked()) and bool(str(details).strip()))
        self.setToolTip(str(tooltip or ""))
        self.title_label.setToolTip(str(tooltip or ""))
        self.value_button.setToolTip(str(tooltip or ""))
        self.details_label.setToolTip(str(tooltip or ""))
