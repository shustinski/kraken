from __future__ import annotations

import secrets
import sys
import threading
import urllib.parse
import urllib.request
from pathlib import Path

from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class DeveloperToolsDialog(QDialog):
    broadcast_result = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Разработчик')
        self.resize(900, 620)
        self._admin_token = secrets.token_urlsafe(32)
        self._process: QProcess | None = None

        root = QVBoxLayout(self)

        form = QFormLayout()
        self.host_edit = QLineEdit('0.0.0.0', self)
        self.port_spin = QSpinBox(self)
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(8000)
        form.addRow('Host', self.host_edit)
        form.addRow('Port', self.port_spin)
        root.addLayout(form)

        server_row = QHBoxLayout()
        self.start_button = QPushButton('Поднять сервер', self)
        self.stop_button = QPushButton('Отключить сервер', self)
        self.stop_button.setEnabled(False)
        self.status_label = QLabel('Сервер остановлен', self)
        server_row.addWidget(self.start_button)
        server_row.addWidget(self.stop_button)
        server_row.addWidget(self.status_label, 1)
        root.addLayout(server_row)

        notify_row = QHBoxLayout()
        self.notification_edit = QLineEdit(self)
        self.notification_edit.setPlaceholderText('Уведомление для всех пользователей WebUI')
        self.send_notification_button = QPushButton('Отправить уведомление', self)
        notify_row.addWidget(self.notification_edit, 1)
        notify_row.addWidget(self.send_notification_button)
        root.addLayout(notify_row)

        self.log_view = QTextEdit(self)
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText('Логи сервера WebUI')
        root.addWidget(self.log_view, 1)

        self.start_button.clicked.connect(self.start_server)
        self.stop_button.clicked.connect(self.stop_server)
        self.send_notification_button.clicked.connect(self.send_notification)
        self.broadcast_result.connect(self._append_log)

    def server_url(self) -> str:
        host = self.host_edit.text().strip() or '127.0.0.1'
        client_host = '127.0.0.1' if host in {'0.0.0.0', '::'} else host
        return f'http://{client_host}:{int(self.port_spin.value())}'

    def start_server(self) -> None:
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            self._append_log('Сервер уже запущен.')
            return

        process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert('NEURALIMAGE_WEBUI_ADMIN_TOKEN', self._admin_token)
        env.insert('PYTHONUNBUFFERED', '1')
        process.setProcessEnvironment(env)
        main_path = Path(__file__).resolve().parent.parent / 'main.py'
        web_args = [
            '--web',
            '--host',
            self.host_edit.text().strip() or '0.0.0.0',
            '--port',
            str(int(self.port_spin.value())),
        ]
        if getattr(sys, 'frozen', False):
            process.setProgram(sys.executable)
            process.setArguments(web_args)
            process.setWorkingDirectory(str(Path(sys.executable).resolve().parent))
        else:
            process.setProgram(sys.executable)
            process.setArguments([str(main_path), *web_args])
            process.setWorkingDirectory(str(main_path.parent))
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.errorOccurred.connect(self._handle_process_error)
        process.finished.connect(self._handle_process_finished)
        self._process = process
        self._append_log(f'Запуск WebUI: {process.program()} {" ".join(process.arguments())}')
        process.start()
        if not process.waitForStarted(5000):
            self._append_log('Не удалось запустить процесс WebUI.')
            self._process = None
            return
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText(f'Сервер запущен: {self.server_url()}')

    def stop_server(self) -> None:
        process = self._process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            self._process = None
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText('Сервер остановлен')
            return
        self._append_log('Остановка WebUI...')
        process.terminate()
        if not process.waitForFinished(5000):
            self._append_log('Процесс не завершился штатно, выполняется kill.')
            process.kill()
            process.waitForFinished(3000)

    def send_notification(self) -> None:
        message = self.notification_edit.text().strip()
        if not message:
            self._append_log('Введите текст уведомления.')
            return
        url = f'{self.server_url()}/api/broadcast/'
        token = self._admin_token
        self.send_notification_button.setEnabled(False)
        threading.Thread(
            target=self._send_notification_worker,
            args=(url, token, message),
            daemon=True,
        ).start()

    def _send_notification_worker(self, url: str, token: str, message: str) -> None:
        try:
            payload = urllib.parse.urlencode(
                {
                    'message': message,
                    'created_by': 'Qt Developer',
                }
            ).encode('utf-8')
            request = urllib.request.Request(
                url,
                data=payload,
                headers={
                    'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
                    'X-NeuralImage-Admin-Token': token,
                },
                method='POST',
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                body = response.read().decode('utf-8', errors='replace')
            self.broadcast_result.emit(f'Уведомление отправлено: {body}')
        except Exception as error:
            self.broadcast_result.emit(f'Не удалось отправить уведомление: {error}')
        finally:
            self.broadcast_result.emit('__enable_notification_button__')

    def _read_stdout(self) -> None:
        process = self._process
        if process is None:
            return
        text = bytes(process.readAllStandardOutput()).decode('utf-8', errors='replace').rstrip()
        if text:
            self._append_log(text)

    def _read_stderr(self) -> None:
        process = self._process
        if process is None:
            return
        text = bytes(process.readAllStandardError()).decode('utf-8', errors='replace').rstrip()
        if text:
            self._append_log(text)

    def _handle_process_error(self, error) -> None:
        self._append_log(f'Ошибка процесса WebUI: {error}')

    def _handle_process_finished(self, exit_code: int, exit_status) -> None:
        self._append_log(f'WebUI завершён: code={exit_code}, status={exit_status}')
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText('Сервер остановлен')

    def _append_log(self, text: str) -> None:
        if text == '__enable_notification_button__':
            self.send_notification_button.setEnabled(True)
            return
        self.log_view.append(str(text))

    def closeEvent(self, event) -> None:
        event.ignore()
        self.hide()

    def shutdown(self) -> None:
        self.stop_server()
