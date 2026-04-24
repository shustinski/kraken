from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QPushButton, QTextBrowser, QVBoxLayout, QWidget

from neuralimage.lib.runtime_paths import resolve_resource_path
from neuralimage.lib.ui_texts import get_ui_section
from neuralimage.lib.version import APP_VERSION


class ChangelogDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        texts = get_ui_section('changelog_dialog')

        self.setWindowTitle(str(texts.get('window_title', 'Список изменений')))
        self.resize(900, 680)

        layout = QVBoxLayout(self)

        text = QTextBrowser(self)
        text.setReadOnly(True)
        text.setOpenExternalLinks(True)
        content = self._resolve_content(str(texts.get('content', '')))
        version_template = str(texts.get('version_template', 'Текущая версия: {version}'))
        text.setMarkdown(f'# {version_template.format(version=APP_VERSION)}\n\n{content}'.strip())
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
                if md_path.parts and md_path.parts[0] == 'resources':
                    md_path = resolve_resource_path(*md_path.parts[1:])
                else:
                    md_path = Path(__file__).resolve().parent.parent / md_path
            if md_path.exists():
                return md_path.read_text(encoding='utf-8')
        return value


def show_changelog_dialog(parent: QWidget | None = None) -> None:
    dialog = ChangelogDialog(parent)
    dialog.exec()
