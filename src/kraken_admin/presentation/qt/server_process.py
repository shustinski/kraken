"""Qt-owned server process and non-blocking readiness checks."""

from __future__ import annotations

import codecs
import json
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, QUrl, pyqtSignal
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest

from kraken_server.configuration import ServerConfig


def server_command(config: Path) -> tuple[str, list[str]]:
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).with_name("KrakenServer.exe")
        if not executable.is_file():
            raise FileNotFoundError(executable)
        return str(executable), ["--config", str(config), "--admin-control"]
    return sys.executable, ["-u", "-m", "kraken_server", "--config", str(config), "--admin-control"]


class ServerProcess(QObject):
    output = pyqtSignal(str)
    changed = pyqtSignal(str)
    stopped = pyqtSignal()
    stopTimedOut = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self.network = QNetworkAccessManager(self)
        self.poll = QTimer(self)
        self.poll.setInterval(500)
        self.poll.timeout.connect(self._probe)
        self.deadline = QTimer(self)
        self.deadline.setSingleShot(True)
        self.deadline.timeout.connect(self._startup_timeout)
        self.stop_deadline = QTimer(self)
        self.stop_deadline.setSingleShot(True)
        self.stop_deadline.timeout.connect(self.stopTimedOut)
        self._reply = None
        self._url = ""
        self._stopping = False
        self._ready_marker = False
        self._tail = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, path: Path) -> None:
        if self.running:
            raise ValueError("Сервер уже запущен")
        config = ServerConfig.load(path)
        program, arguments = server_command(config.path.resolve())
        host = config.host
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1" if host == "0.0.0.0" else "[::1]"
        elif ":" in host and not host.startswith("["):
            host = f"[{host}]"
        self._url = f"{'https' if config.tls_cert_file else 'http'}://{host}:{config.port}/api/v1/health"
        self._stopping = False
        self._ready_marker = False
        self._tail = ""
        self._decoder.reset()
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(environment)
        self.process.start(program, arguments)
        self.changed.emit(f"Запуск: {self._url}")
        self.poll.start()
        self.deadline.start(30000)

    def stop(self) -> None:
        if self.running and not self._stopping:
            self._stopping = True
            self.poll.stop()
            self.deadline.stop()
            self.changed.emit("Остановка…")
            self.process.write(b"stop\n")
            self.stop_deadline.start(15000)

    def kill(self) -> None:
        if self.running:
            self._stopping = True
            self.process.kill()

    def _read(self) -> None:
        value = self._decoder.decode(bytes(self.process.readAllStandardOutput()))
        self._tail = (self._tail + value)[-4096:]
        self._ready_marker = self._ready_marker or "KRAKEN_ADMIN_READY" in self._tail
        self.output.emit(value)

    def _probe(self) -> None:
        if not self._ready_marker or self._reply is not None or not self.running:
            return
        request = QNetworkRequest(QUrl(self._url))
        request.setTransferTimeout(2000)
        reply = self.network.get(request)
        self._reply = reply

        def complete() -> None:
            self._reply = None
            try:
                body = json.loads(bytes(reply.readAll()).decode("utf-8"))
            except (ValueError, UnicodeError):
                body = {}
            if body.get("status") == "ok" and self.running and not self._stopping:
                self.poll.stop()
                self.deadline.stop()
                self.changed.emit(f"Работает: {self._url}")
            reply.deleteLater()

        reply.finished.connect(complete)

    def _startup_timeout(self) -> None:
        self.output.emit("\nСервер не подтвердил готовность за 30 секунд. Проверьте журнал и TLS.\n")
        self.stop()

    def _error(self, _error: QProcess.ProcessError) -> None:
        self.output.emit(self.process.errorString() + "\n")
        if not self.running:
            self._finished(-1)

    def _finished(self, code: int, *_args: object) -> None:
        self._read()
        self.poll.stop()
        self.deadline.stop()
        self.stop_deadline.stop()
        if self._reply is not None:
            self._reply.abort()
        self.changed.emit("Остановлен" if self._stopping else f"Процесс завершён, код {code}")
        self.stopped.emit()
