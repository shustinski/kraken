from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QPushButton, QTextEdit, QVBoxLayout, QWidget

from lib.ui_texts import get_ui_section


class HelpDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        texts = get_ui_section('help_dialog')

        self.setWindowTitle(str(texts.get('window_title', 'Справка')))
        self.resize(980, 760)

        layout = QVBoxLayout(self)

        text = QTextEdit(self)
        text.setReadOnly(True)
        text.setAcceptRichText(False)
        text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        text.setPlainText(str(texts.get('content', '')))
        text.moveCursor(text.textCursor().MoveOperation.Start)
        layout.addWidget(text)

        close_button = QPushButton(str(texts.get('close_button', 'Закрыть')), self)
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)


def show_help_dialog(parent: QWidget | None = None) -> None:
    dialog = HelpDialog(parent)
    dialog.exec()
