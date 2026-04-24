from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import QLabel, QSizePolicy


class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._full_text = ''
        self._base_tooltip = ''
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWordWrap(False)

    def setText(self, text: str) -> None:
        self._full_text = str(text or '')
        self._update_display_text()
        self._update_tooltip()

    def text(self) -> str:
        return str(self._full_text)

    def setToolTip(self, text: str) -> None:
        self._base_tooltip = str(text or '')
        self._update_tooltip()

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._update_display_text()
        super().resizeEvent(event)

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())

    def mousePressEvent(self, QMouseEvent):
        self.clicked.emit()
        QLabel.mousePressEvent(self, QMouseEvent)

    def _update_display_text(self) -> None:
        full_text = str(self._full_text)
        if not full_text:
            QLabel.setText(self, '')
            return
        available_width = max(0, self.contentsRect().width())
        if available_width <= 0:
            QLabel.setText(self, full_text)
            return
        elided = self.fontMetrics().elidedText(
            full_text,
            Qt.TextElideMode.ElideMiddle,
            available_width,
        )
        QLabel.setText(self, elided)

    def _update_tooltip(self) -> None:
        parts: list[str] = []
        base_tooltip = str(self._base_tooltip).strip()
        full_text = str(self._full_text).strip()
        if base_tooltip:
            parts.append(base_tooltip)
        if full_text and full_text != base_tooltip:
            parts.append(full_text)
        QLabel.setToolTip(self, '\n\n'.join(parts))
