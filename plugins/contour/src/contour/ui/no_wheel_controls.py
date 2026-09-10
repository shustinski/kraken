"""Combo/spin widgets that ignore mouse-wheel value changes on hover."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox


class NoWheelComboBox(QComboBox):
    """Change the current item only from the popup or clicks, never from the wheel."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is None:
            return
        view = self.view()
        if view is not None and view.isVisible():
            super().wheelEvent(event)
            return
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """Keep the value unchanged when the cursor is over the field and the wheel moves."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is not None:
            event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Keep the value unchanged when the cursor is over the field and the wheel moves."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is not None:
            event.ignore()
