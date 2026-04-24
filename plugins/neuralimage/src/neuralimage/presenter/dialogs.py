from __future__ import annotations

from PyQt6 import QtCore
from PyQt6.QtGui import QTextDocument
from PyQt6.QtWidgets import QFileDialog, QMessageBox


def format_auto_answer_button_text(text: str, seconds_left: int) -> str:
    if seconds_left <= 0:
        return text
    return f'{text} ({int(seconds_left)})'


def choose_path_dialog(kind: str, filetypes=None) -> str | None:
    if kind == 'folder':
        path = QFileDialog.getExistingDirectory(None, 'Выберите папку')
    else:
        filter_str = ''
        if filetypes:
            labels = []
            for title, ext in filetypes:
                cleaned = ext if str(ext).startswith('*.') else f'*{ext}'
                labels.append(f'{title} ({cleaned})')
            filter_str = ';;'.join(labels)
        path, _ = QFileDialog.getOpenFileName(None, 'Выберите файл', '', filter_str)
    return path if path else None


def ask_yes_no_with_timeout(
    *,
    parent,
    question: str,
    header: str,
    default_answer: bool,
    timeout_seconds: int,
) -> bool:
    buttons = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    default_button = QMessageBox.StandardButton.Yes if default_answer else QMessageBox.StandardButton.No

    dialog = QMessageBox(parent)
    dialog.setIcon(QMessageBox.Icon.Question)
    dialog.setWindowTitle(header)
    dialog.setText(question)
    dialog.setStandardButtons(buttons)
    dialog.setDefaultButton(default_button)
    dialog.setEscapeButton(default_button)

    if timeout_seconds > 0:
        timer = QtCore.QTimer(dialog)
        timer.setInterval(1000)
        seconds_left = max(0, int(timeout_seconds))
        default_button_widget = dialog.button(default_button)
        default_button_text = default_button_widget.text() if default_button_widget is not None else ''

        def _update_button_text() -> None:
            if default_button_widget is None:
                return
            default_button_widget.setText(
                format_auto_answer_button_text(default_button_text, seconds_left)
            )

        def _auto_answer() -> None:
            nonlocal seconds_left
            if not dialog.isVisible():
                return
            seconds_left -= 1
            if seconds_left <= 0:
                timer.stop()
                if default_button_widget is not None:
                    default_button_widget.setText(default_button_text)
                    default_button_widget.click()
                else:
                    dialog.done(int(default_button))
                return
            _update_button_text()

        _update_button_text()
        timer.timeout.connect(_auto_answer)
        dialog.finished.connect(timer.stop)
        timer.start()

    dialog.exec()
    clicked_button = dialog.clickedButton()
    reply = dialog.standardButton(clicked_button) if clicked_button is not None else default_button
    return reply == QMessageBox.StandardButton.Yes


def markdown_to_message_html(markdown: str) -> str:
    document = QTextDocument()
    document.setMarkdown(str(markdown or '').strip())
    return document.toHtml()
