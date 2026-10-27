from __future__ import annotations

from kraken_hub.agent_runtime import LocalAgentRuntime


def test_source_plugin_can_launch_without_package_installation(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    from kraken_core.plugins import PluginInventoryItem, load_plugin_metadata_from_directory

    root = Path(__file__).resolve().parents[1]
    plugin = load_plugin_metadata_from_directory(root / "plugins/contour")
    runtime = LocalAgentRuntime(tmp_path / "agent", (PluginInventoryItem(plugin, installed=False),))
    registry = runtime.interactive_registry("frames.vectorize.v1")
    spec = registry.get("frames.vectorize.v1")
    assert spec.command == (sys.executable, str(root / "plugins/contour/__main__.py"))
    result = subprocess.run([*spec.command, "--help"], cwd=spec.working_directory, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def test_local_agent_runtime_starts_lazily_and_shuts_down_cleanly(tmp_path) -> None:
    runtime = LocalAgentRuntime(tmp_path / "agent")
    assert runtime.process is None
    assert runtime.base_url == ""

    try:
        assert runtime.ensure_started() == frozenset()
        process = runtime.process
        assert process is not None
        assert process.poll() is None
        assert runtime.base_url.startswith("http://127.0.0.1:")
        assert runtime.token
        assert runtime._request("GET", "/api/v1/health")["status"] == "ok"

        assert runtime.ensure_started() == frozenset()
        assert runtime.process is process
    finally:
        runtime.shutdown()

    assert process.poll() == 0
    assert runtime.base_url == ""
    assert runtime.token == ""
    assert not tuple((tmp_path / "agent").glob("connection-*.json"))


def test_concurrent_scheduler_and_ui_share_one_agent_process(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import kraken_hub.agent_runtime as runtime_module

    started = []
    launch = runtime_module.subprocess.Popen

    def tracked_launch(*args, **kwargs):
        process = launch(*args, **kwargs)
        started.append(process.pid)
        return process

    monkeypatch.setattr(runtime_module.subprocess, "Popen", tracked_launch)

    runtime = LocalAgentRuntime(tmp_path / "agent")
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            values = list(executor.map(lambda _: runtime.ensure_started(), range(2)))
        assert values == [frozenset(), frozenset()]
        assert runtime.process.poll() is None
        assert started == [runtime.process.pid]
        assert not tuple((tmp_path / "agent").glob("connection-*.json"))
    finally:
        runtime.shutdown()


def test_desktop_private_agent_entry_runs_the_real_control_service(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys
    import time
    from urllib.request import Request, urlopen

    connection = tmp_path / "connection.json"
    root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [
            sys.executable,
            str(root / "packaging/entries/desktop_entry.py"),
            "--kraken-agent",
            "--data-dir",
            str(tmp_path / "agent"),
            "--connection-file",
            str(connection),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 15
        while not connection.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert connection.exists()
        config = json.loads(connection.read_text(encoding="utf-8"))
        headers = {"Authorization": f"Bearer {config['token']}"}
        with urlopen(Request(config["url"] + "/api/v1/health", headers=headers), timeout=5) as response:
            assert json.load(response)["status"] == "ok"
        with urlopen(
            Request(config["url"] + "/api/v1/shutdown", data=b"{}", headers=headers, method="POST"), timeout=5
        ):
            pass
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
