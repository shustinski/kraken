from __future__ import annotations

from kraken_hub.infrastructure.agent.runtime import LocalAgentRuntime


def test_agent_survives_hub_exit_and_reconnects(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import textwrap
    import time

    data_dir = tmp_path / "agent"
    started = tmp_path / "plugin-started"
    release = tmp_path / "plugin-release"
    plugin = tmp_path / "plugin.py"
    plugin.write_text(textwrap.dedent(f"""
        import os
        import time
        from pathlib import Path
        from kraken_core.process_lifetime import bind_child_process_lifetime
        from kraken_core.plugin_protocol import PluginResultManifest
        bind_child_process_lifetime()
        Path({str(started)!r}).touch()
        deadline = time.monotonic() + 20
        while not Path({str(release)!r}).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("test did not release plugin")
            time.sleep(0.05)
        result = PluginResultManifest(job_id="job-1", outcome="partial",
                                      plugin_id="test", plugin_version="1", outputs=())
        Path(os.environ["KRAKEN_RESULT_MANIFEST"]).write_text(result.to_json())
    """), encoding="utf-8")
    script = textwrap.dedent(f"""
        import hashlib
        import json
        import sys
        import time
        from pathlib import Path
        from kraken_agent.jobs import DurableJobStore
        from kraken_core.plugin_protocol import PluginJobManifest, PluginFrameInput
        from kraken_core.process_lifetime import bind_child_process_lifetime
        from kraken_hub.infrastructure.agent.runtime import LocalAgentRuntime
        bind_child_process_lifetime()
        class Runtime(LocalAgentRuntime):
            def _registry(self):
                path = self.data_dir / "test-registry.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({{"plugins": [{{"operation": "test.v1",
                    "command": [sys.executable, {str(plugin)!r}], "interactive": True}}]}}))
                return path, frozenset()
        runtime = Runtime({str(data_dir)!r})
        runtime.ensure_started()
        source = runtime.data_dir / "staging/job-1/inputs/1.png"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"input")
        manifest = PluginJobManifest(job_id="job-1", operation="test.v1",
            project_id="project-1", layer_id="layer-1", actor_id="actor-1",
            target_representation_id="representation-1", inputs=(PluginFrameInput(
                "frame-1", 1, 1, "version-1", hashlib.sha256(b"input").hexdigest(),
                "image/png", "inputs/1.png"),))
        DurableJobStore(runtime.data_dir / "jobs.sqlite3").enqueue(manifest)
        deadline = time.monotonic() + 10
        while not Path({str(started)!r}).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("plugin did not start")
            time.sleep(0.05)
        runtime.detach()
    """)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    runtime = LocalAgentRuntime(data_dir)
    try:
        completed = subprocess.run([sys.executable, "-c", script], env=env,
                                   capture_output=True, text=True, timeout=30)
        assert completed.returncode == 0, completed.stderr
        connection = json.loads((data_dir / "runtime-connection.json").read_text())
        assert runtime.ensure_started() == frozenset()
        assert runtime.base_url == connection["url"]
        assert runtime.token == connection["token"]
        assert runtime.process is None  # attached to the existing process
        assert runtime._request("GET", "/api/v1/health")["status"] == "ok"
        assert runtime._request("GET", "/api/v1/jobs/job-1")["state"] == "waiting_for_user"
        release.touch()
        deadline = time.monotonic() + 10
        while runtime._request("GET", "/api/v1/jobs/job-1")["state"] == "waiting_for_user":
            assert time.monotonic() < deadline, "plugin did not complete after Hub exited"
            time.sleep(0.05)
        assert runtime._request("GET", "/api/v1/jobs/job-1/result")["outcome"] == "partial"
    finally:
        runtime.shutdown()


def test_stale_agent_connection_is_replaced(tmp_path):
    import json

    data_dir = tmp_path / "agent"
    data_dir.mkdir()
    saved = data_dir / "runtime-connection.json"
    saved.write_text(json.dumps({"url": "http://127.0.0.1:0", "token": "stale"}))
    runtime = LocalAgentRuntime(data_dir)
    try:
        runtime.ensure_started()
        assert runtime.process.poll() is None
        assert runtime.token != "stale"
    finally:
        runtime.shutdown()


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
    import kraken_hub.infrastructure.agent.runtime as runtime_module

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
