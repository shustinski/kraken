"""main_window: correlation column widget."""
from __future__ import annotations
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

class _CorrelationColumnWidget(QFrame):
    """Show one colored correlation column as one clickable filter block."""

    columnClicked = pyqtSignal(str)

    def __init__(self, title: str, band: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.band = str(band)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.header_button = QPushButton(title, self)
        self.header_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header_button.clicked.connect(lambda: self.columnClicked.emit(self.band))
        self.summary_label = QLabel('-', self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.summary_label.setStyleSheet('color: #c9d3df; background: transparent; border: none;')
        self.summary_label.setVisible(False)
        layout.addWidget(self.header_button)
        layout.addStretch(1)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._refresh_style(False)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.columnClicked.emit(self.band)
            event.accept()
            return
        super().mousePressEvent(event)

    def _refresh_style(self, active: bool) -> None:
        palette = ('#8c2f39', '#ffe9ec', '#b84a58') if self.band == 'bad' else ('#1f5f3b', '#e9fff1', '#2d7a50')
        border_width = '2px' if active else '1px'
        self.setStyleSheet(
            'QFrame {'
            f'background-color: {palette[0]}; color: {palette[1]}; border: {border_width} solid {palette[2]}; '
            'border-radius: 10px;'
            '}'
            'QLabel { color: #f4f7fb; background: transparent; border: none; }'
        )
        self.header_button.setStyleSheet(
            'QPushButton {'
            f'background-color: rgba(0, 0, 0, 0.10); color: {palette[1]}; border: 0px; '
            'border-radius: 8px; padding: 8px 10px; font-weight: 700; text-align: left;'
            '}'
        )

    def set_payload(self, frame_count: int, mean_hits: float, max_hits: int, *, active: bool) -> None:
        self._refresh_style(active)
        self.summary_label.clear()
