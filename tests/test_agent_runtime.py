from __future__ import annotations

from kraken_hub.agent_runtime import LocalAgentRuntime


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
