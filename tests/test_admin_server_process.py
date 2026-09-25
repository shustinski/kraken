from __future__ import annotations

import socket
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

from kraken_admin.server_process import ServerProcess, server_command


def test_packaged_and_source_command(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    executable, arguments = server_command(tmp_path / "server.toml")
    assert executable == sys.executable
    assert "--admin-control" in arguments
    assert arguments[2] == "kraken_server"
    monkeypatch.setattr(sys, "frozen", True)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "KrakenAdmin.exe"))
    with pytest.raises(FileNotFoundError):
        server_command(tmp_path / "server.toml")
    (tmp_path / "KrakenServer.exe").touch()
    assert server_command(tmp_path / "server.toml")[0] == str(tmp_path / "KrakenServer.exe")


def test_server_readiness_logs_stop_and_restart(qtbot, monkeypatch, tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    config = SimpleNamespace(path=tmp_path / "server.toml", host="127.0.0.1", port=port, tls_cert_file=None)
    monkeypatch.setattr("kraken_admin.server_process.ServerConfig.load", lambda _: config)
    script = (
        "import uvicorn\n"
        "from fastapi import FastAPI\n"
        "from kraken_server.admin_control import run_controlled\n"
        "app=FastAPI()\n"
        "@app.get('/api/v1/health')\n"
        "def health(): return {'status':'ok'}\n"
        f"run_controlled(uvicorn.Config(app,host='127.0.0.1',port={port}))\n"
    )
    monkeypatch.setattr("kraken_admin.server_process.server_command", lambda _: (sys.executable, ["-u", "-c", script]))
    runtime = ServerProcess()
    output, states = [], []
    runtime.output.connect(output.append)
    runtime.changed.connect(states.append)
    try:
        for _ in range(2):
            runtime.start(config.path)
            qtbot.waitUntil(lambda: states[-1].startswith("Работает:"), timeout=15000)
            with pytest.raises(ValueError, match="уже"):
                runtime.start(config.path)
            runtime.stop()
            qtbot.waitUntil(lambda: not runtime.running, timeout=10000)
        assert "KRAKEN_ADMIN_READY" in "".join(output)
        assert "Application shutdown complete" in "".join(output)
    finally:
        if runtime.running:
            runtime.kill()
            qtbot.waitUntil(lambda: not runtime.running, timeout=5000)


def test_failed_start_reports_exit(qtbot, monkeypatch, tmp_path):
    config = SimpleNamespace(path=tmp_path / "server.toml", host="127.0.0.1", port=1, tls_cert_file=None)
    monkeypatch.setattr("kraken_admin.server_process.ServerConfig.load", lambda _: config)
    monkeypatch.setattr("kraken_admin.server_process.server_command",
                        lambda _: (sys.executable, ["-c", "print('port unavailable'); raise SystemExit(3)"]))
    runtime = ServerProcess()
    states, output = [], []
    runtime.changed.connect(states.append)
    runtime.output.connect(output.append)
    runtime.start(config.path)
    qtbot.waitUntil(lambda: any("код 3" in state for state in states), timeout=10000)
    assert "port unavailable" in "".join(output)
    assert not runtime.poll.isActive()
