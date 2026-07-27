"""Label with a button-like click signal and path-friendly text elision."""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QResizeEvent
from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget


class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = ""
        self._base_tooltip = ""
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWordWrap(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setText(text)

    def setText(self, text: str) -> None:
        self._full_text = str(text or "")
        self._update_display_text()
        self._update_tooltip()

    def text(self) -> str:
        return self._full_text

    def setToolTip(self, text: str) -> None:
        self._base_tooltip = str(text or "")
        self._update_tooltip()

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._update_display_text()
        super().resizeEvent(event)

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.clicked.emit()
        super().mousePressEvent(event)

    def _update_display_text(self) -> None:
        if not self._full_text:
            QLabel.setText(self, "")
            return
        width = max(0, self.contentsRect().width())
        visible = (
            self.fontMetrics().elidedText(self._full_text, Qt.TextElideMode.ElideMiddle, width)
            if width
            else self._full_text
        )
        QLabel.setText(self, visible)

    def _update_tooltip(self) -> None:
        parts = [value for value in (self._base_tooltip.strip(), self._full_text.strip()) if value]
        QLabel.setToolTip(self, "\n\n".join(dict.fromkeys(parts)))


__all__ = ["ClickableLabel"]
