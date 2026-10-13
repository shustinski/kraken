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
