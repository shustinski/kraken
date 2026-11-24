"""Protocol and recording checks for the external Inspector process."""

from __future__ import annotations

import io
import json
from pathlib import Path

from kraken_inspector.mcp import InspectorMCP, serve
from kraken_inspector.session import InspectorSession


def _frame(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8") + b"\n"


def _responses(data: bytes) -> list[dict]:
    results = []
    stream = io.BytesIO(data)
    while line := stream.readline():
        results.append(json.loads(line))
    return results


def test_mcp_initializes_and_discovers_tools(tmp_path: Path) -> None:
    service = InspectorMCP(InspectorSession(tmp_path))
    stream = io.BytesIO(
        _frame({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
        + _frame({"jsonrpc": "2.0", "method": "notifications/initialized"})
        + _frame({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    )
    output = io.BytesIO()
    serve(service, stream, output)
    initialized, listed = _responses(output.getvalue())
    assert initialized["result"]["serverInfo"]["name"] == "kraken-inspector"
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert {"launch_hub", "inspect_ui", "perform_action", "replay_scenario", "explore_states"} <= names


def test_recorded_scenario_preserves_action_and_revision(tmp_path: Path) -> None:
    session = InspectorSession(tmp_path)
    session.record_action("w0:0/1", "click", "", {"ok": True})
    path = session.save_scenario()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["actions"][0]["id"] == "w0:0/1"
    assert data["actions"][0]["result"]["ok"]
    assert "revision" in data
    assert path.parent == tmp_path / ".tmp-tests" / "inspector"
