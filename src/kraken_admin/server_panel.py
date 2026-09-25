"""Server controls embedded in the local administration window."""

from pathlib import Path

from PyQt6.QtCore import QObject, QEvent
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from .server_process import ServerProcess


class ServerPanel(QWidget):
    def __init__(self, admin) -> None:
        super().__init__(admin.window)
        self.admin = admin
        self.runtime = ServerProcess(self)
        self.closing = False
        layout = QVBoxLayout(self)
        self.status = QLabel("Остановлен")
        layout.addWidget(self.status)
        actions = QHBoxLayout()
        self.start_button = QPushButton("Запустить")
        self.stop_button = QPushButton("Остановить")
        self.stop_button.setEnabled(False)
        save = QPushButton("Сохранить журнал…")
        for button in (self.start_button, self.stop_button, save):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(10000)
        layout.addWidget(self.log)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.runtime.stop)
        save.clicked.connect(self.save)
        self.runtime.output.connect(self.append)
        self.runtime.changed.connect(self.changed)
        self.runtime.stopped.connect(self.finished)
        self.runtime.stopTimedOut.connect(self.force_stop)
        self.filter = _CloseFilter(self)
        admin.window.installEventFilter(self.filter)

    def start(self) -> None:
        try:
            if self.admin._config_path is None:
                raise ValueError("Сначала выберите server.toml")
            self.runtime.start(self.admin._config_path)
        except Exception as exc:
            self.admin._report("Не удалось запустить сервер", exc)

    def changed(self, status: str) -> None:
        self.status.setText(status)
        self.start_button.setEnabled(not self.runtime.running)
        self.stop_button.setEnabled(self.runtime.running)

    def append(self, value: str) -> None:
        bar = self.log.verticalScrollBar()
        follow = bar.value() >= bar.maximum() - 2
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(value)
        if follow:
            bar.setValue(bar.maximum())

    def save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить журнал", "server.log", "Log (*.log)")
        if path:
            try:
                Path(path).write_text(self.log.toPlainText(), encoding="utf-8")
            except OSError as exc:
                self.admin._report("Не удалось сохранить журнал", exc)

    def force_stop(self) -> None:
        if QMessageBox.question(self, "Сервер не остановился",
                                "Принудительно завершить принадлежащий Admin процесс?") == QMessageBox.StandardButton.Yes:
            self.runtime.kill()
        else:
            self.closing = False

    def finished(self) -> None:
        if self.closing:
            self.admin.window.close()


class _CloseFilter(QObject):
    def __init__(self, panel: ServerPanel) -> None:
        super().__init__(panel)
        self.panel = panel

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Close:
            if getattr(self.panel.admin, "_deletion_busy", False):
                event.ignore()
                self.panel.admin._report("Удаление выполняется", "Дождитесь завершения текущего шага очистки.")
                return True
            if self.panel.runtime.running:
                event.ignore()
                if not self.panel.closing and QMessageBox.question(
                    self.panel, "Закрыть Kraken Admin", "Остановить сервер и закрыть окно?",
                ) == QMessageBox.StandardButton.Yes:
                    self.panel.closing = True
                    self.panel.runtime.stop()
                return True
        return super().eventFilter(watched, event)
