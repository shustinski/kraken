"""Палитра W3: HTTP health локального агента и защита токеном. S09."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest


def test_s09_agent_health_ok_with_token_and_rejected_without(tmp_path):
    filename = os.environ.get("KRAKEN_LIVE_WORKER_CONFIG")
    if not filename:
        pytest.skip("Нужен выделенный beta-стенд для проверки S09")
    stand = json.loads(Path(filename).read_text(encoding="utf-8-sig"))
    connection = tmp_path / "connection.json"
    process = subprocess.Popen(
        [sys.executable, "-m", "kraken_agent", "--independent", "--data-dir", str(tmp_path / "agent"),
         "--connection-file", str(connection)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 15
        while not connection.is_file() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert connection.is_file(), f"agent exited with {process.poll()} before writing connection file"
        config = json.loads(connection.read_text(encoding="utf-8"))
        health_url = config["url"] + "/api/v1/health"
        headers = {"Authorization": f"Bearer {config['token']}"}
        with urlopen(Request(health_url, headers=headers), timeout=5) as response:
            assert json.load(response)["status"] == "ok"
        with pytest.raises(HTTPError) as denied:
            urlopen(health_url, timeout=5)
        assert denied.value.code == 401
        with urlopen(Request(config["url"] + "/api/v1/shutdown", data=b"{}", headers=headers, method="POST"), timeout=5):
            pass
        assert process.wait(timeout=10) == 0
        with pytest.raises(URLError):
            urlopen(health_url, timeout=1)
        with urlopen(stand["server_url"] + "/api/v1/health", timeout=5) as response:
            assert json.load(response)["status"] == "ok"
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
