"""Execution destination controls and connection checks."""
from __future__ import annotations
import os
import json
from pathlib import Path
from PyQt6.QtCore import QSettings, QThread, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QLineEdit, QPushButton, QToolBar, QMessageBox, QFileDialog
from neuralimage.lib.ui_texts import get_ui_language
from .client import RemoteClient

LABELS = {
    "execution": ("Execution", "Выполнение"),
    "local": ("On this computer", "На этом компьютере"),
    "remote": ("On server", "На сервере"),
    "address": ("NeuralImage server URL on the local network", "Адрес сервера NeuralImage в локальной сети"),
    "check": ("Check connection", "Проверить соединение"),
    "restore": ("Restore remote job", "Восстановить удалённую задачу"),
    "record": ("Remote job record", "Файл удалённой задачи"),
    "server": ("NeuralImage server", "Сервер NeuralImage"),
    "connected": ("Connected. Protocol is compatible.", "Соединение установлено. Протокол совместим."),
}


def text(key: str, language: str | None = None) -> str:
    return LABELS[key][1 if (language or get_ui_language()) == "ru" else 0]


def remote_url() -> str | None:
    settings = QSettings()
    if os.environ.get("NEURALIMAGE_REMOTE_ONLY") or settings.value("execution/mode", "local") == "remote":
        return str(settings.value("execution/url", "http://localhost:8765"))
    return None


class ConnectionCheck(QThread):
    checked = pyqtSignal(str)
    def __init__(self, url: str):
        super().__init__()
        self.url = url
    def run(self) -> None:
        try:
            RemoteClient(self.url, timeout=5).capabilities()
            self.checked.emit("")
        except Exception as error:
            self.checked.emit(str(error))


def install_remote_controls(presenter) -> None:
    toolbar = QToolBar(text("execution"), presenter.view)
    mode = QComboBox()
    mode.addItem(text("local"), "local")
    mode.addItem(text("remote"), "remote")
    settings = QSettings()
    mode.setCurrentIndex(1 if remote_url() is not None else 0)
    mode.setEnabled(not bool(os.environ.get("NEURALIMAGE_REMOTE_ONLY")))
    address = QLineEdit(str(settings.value("execution/url", "http://localhost:8765")))
    address.setPlaceholderText("http://server:8765")
    address.setToolTip(text("address"))
    check = QPushButton(text("check"))
    recover = QPushButton(text("restore"))
    for widget in (mode, address, check, recover):
        toolbar.addWidget(widget)
    presenter.view.addToolBar(toolbar)
    def retranslate(language):
        toolbar.setWindowTitle(text("execution", language))
        mode.setItemText(0, text("local", language))
        mode.setItemText(1, text("remote", language))
        address.setToolTip(text("address", language))
        check.setText(text("check", language))
        recover.setText(text("restore", language))
    presenter.view.ui_language_selected.connect(retranslate)
    mode.currentIndexChanged.connect(lambda: settings.setValue("execution/mode", mode.currentData()))
    address.editingFinished.connect(lambda: settings.setValue("execution/url", address.text().strip()))
    def restore():
        filename, _ = QFileDialog.getOpenFileName(presenter.view, text("record"), "", "JSON (*.json)")
        if not filename:
            return
        try:
            from neuralimage.application.dto import MainWindowState, SettingsState
            value = json.loads(Path(filename).read_text(encoding="utf-8"))
            task = presenter._processing_session.enqueue_task(MainWindowState(**value["main"]), SettingsState(**value["settings"]))
            task.runtime.execution_mode = "remote"
            task.runtime.remote_job_id = value["job"]
            task.runtime.remote_url = value["url"]
            task.runtime.remote_request_key = value["request_key"]
            presenter._refresh_queue_view(selected_task_id=task.task_id)
            presenter._start_next_task_if_possible()
        except (OSError, ValueError, TypeError, KeyError) as error:
            QMessageBox.warning(presenter.view, text("restore"), str(error))
    recover.clicked.connect(restore)
    def start_check():
        settings.setValue("execution/url", address.text().strip())
        check.setEnabled(False)
        thread = ConnectionCheck(address.text().strip())
        presenter._remote_connection_check = thread
        thread.checked.connect(lambda error: QMessageBox.information(presenter.view, text("server"), error or text("connected")))
        thread.finished.connect(lambda: check.setEnabled(True))
        thread.start()
    check.clicked.connect(start_check)
