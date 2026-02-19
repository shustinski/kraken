from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QPushButton, QTextEdit, QVBoxLayout, QWidget

from lib.ui_texts import get_ui_section
from lib.version import APP_VERSION


class ChangelogDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        texts = get_ui_section('changelog_dialog')

        self.setWindowTitle(str(texts.get('window_title', 'Список изменений')))
        self.resize(900, 680)

        layout = QVBoxLayout(self)

        text = QTextEdit(self)
        text.setReadOnly(True)
        text.setAcceptRichText(False)
        text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        content = self._resolve_content(str(texts.get('content', '')))
        text.setPlainText(f'Текущая версия: {APP_VERSION}\n\n{content}')
        text.moveCursor(text.textCursor().MoveOperation.Start)
        layout.addWidget(text)

        close_button = QPushButton(str(texts.get('close_button', 'Закрыть')), self)
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _resolve_content(content: str) -> str:
        value = str(content or '').strip()
        if not value:
            return ''
        if value.lower().endswith('.md'):
            md_path = Path(value)
            if not md_path.is_absolute():
                root = Path(__file__).resolve().parent.parent
                md_path = root / md_path
            if md_path.exists():
                return md_path.read_text(encoding='utf-8')
        return value


def show_changelog_dialog(parent: QWidget | None = None) -> None:
    dialog = ChangelogDialog(parent)
    dialog.exec()
