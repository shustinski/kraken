"""main_window: expandable score card."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

class _ExpandableScoreCard(QWidget):
    """Show one score as a clickable card with expandable details."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.title_label = QLabel(title, self)
        self.title_label.setWordWrap(True)
        value_row = QWidget(self)
        value_layout = QHBoxLayout(value_row)
        value_layout.setContentsMargins(0, 0, 0, 0)
        value_layout.setSpacing(6)
        self.value_button = QPushButton('-', value_row)
        self.value_button.setCheckable(True)
        self.value_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.value_button.setStyleSheet('padding: 6px 10px; border-radius: 8px; background-color: #2f3844; color: #edf3fb; font-weight: 700; border: none; text-align: center;')
        self.percentile_label = QLabel('-', value_row)
        self.percentile_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.percentile_label.setMinimumWidth(72)
        self.percentile_label.setStyleSheet('padding: 6px 10px; border-radius: 8px; background-color: #2f3844; color: #edf3fb; font-weight: 700;')
        self.details_label = QLabel('', self)
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet('padding: 4px 6px; color: #c9d3df; background-color: #11161d; border-radius: 6px;')
        self.details_label.hide()
        self.value_button.toggled.connect(self._on_toggled)
        value_layout.addWidget(self.value_button, stretch=1)
        value_layout.addWidget(self.percentile_label)
        layout.addWidget(self.title_label)
        layout.addWidget(value_row)
        layout.addWidget(self.details_label)

    def _on_toggled(self, checked: bool) -> None:
        self.details_label.setVisible(bool(checked) and bool(self.details_label.text().strip()))

    def set_payload(self, text: str, style: str, details: str, *, visible: bool, percentile_text: str = '-', percentile_style: str | None = None, tooltip: str = '') -> None:
        self.setVisible(visible)
        self.value_button.setText(text)
        self.value_button.setStyleSheet(style + '; border: none; text-align: center;')
        self.percentile_label.setText(percentile_text)
        self.percentile_label.setStyleSheet((percentile_style or style) + ';')
        self.percentile_label.setVisible(visible and bool(percentile_text.strip()))
        self.details_label.setText(details)
        self.details_label.setVisible(bool(self.value_button.isChecked()) and bool(details.strip()) and visible)
        self.setToolTip(tooltip)
        self.title_label.setToolTip(tooltip)
        self.value_button.setToolTip(tooltip)
        self.percentile_label.setToolTip(tooltip)
        self.details_label.setToolTip(tooltip)

class _NoWheelSpinBox(QSpinBox):
    """Ignore wheel scrolling to avoid accidental value changes inside the control panel."""

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()

class _NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Ignore wheel scrolling to avoid accidental value changes inside the control panel."""

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()

class _NoWheelComboBox(QComboBox):
    """Ignore wheel scrolling unless the popup list is explicitly open."""

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        view = self.view()
        if view is not None and view.isVisible():
            super().wheelEvent(event)
            return
        event.ignore()
