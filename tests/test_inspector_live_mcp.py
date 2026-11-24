"""Exercise MCP requests through stdio and a real isolated Hub process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def test_mcp_client_launches_and_clicks_real_hub() -> None:
    workspace = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    server = subprocess.Popen(
        [sys.executable, "-m", "kraken_inspector", "--workspace", str(workspace)],
        cwd=workspace,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert server.stdin is not None and server.stdout is not None

    def call(request_id: int, method: str, params: dict | None = None) -> dict:
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        server.stdin.write(json.dumps(payload).encode() + b"\n")
        server.stdin.flush()
        return json.loads(server.stdout.readline())

    try:
        initialized = call(1, "initialize", {"protocolVersion": "2025-06-18"})
        assert initialized["result"]["capabilities"]["tools"] == {"listChanged": False}
        assert "launch_hub" in {item["name"] for item in call(2, "tools/list")["result"]["tools"]}
        launched = call(3, "tools/call", {"name": "launch_hub", "arguments": {"timeout": 30}})
        assert not launched["result"].get("isError"), launched
        ui = call(4, "tools/call", {"name": "inspect_ui", "arguments": {}})["result"]["structuredContent"]
        assert any(window["title"] == "Kraken" for window in ui["windows"])
        button = next(
            node
            for window in ui["windows"]
            for node in window["nodes"]
            if node["class"] == "QPushButton" and node["text"] == "Создать проект"
        )
        action = call(5, "tools/call", {"name": "perform_action", "arguments": {"id": button["id"], "action": "click"}})
        assert action["result"]["structuredContent"]["ok"]
        assert not action["result"]["structuredContent"]["invariants"]["violations"]
        waited = call(
            17, "tools/call", {"name": "wait_for_ui", "arguments": {"window_title": "Новый проект", "timeout": 5}}
        )
        assert waited["result"]["structuredContent"]["found"]
        scenario = call(6, "tools/call", {"name": "save_scenario", "arguments": {}})
        assert Path(scenario["result"]["structuredContent"]["path"]).is_file()
        monitoring = call(7, "tools/call", {"name": "start_monitoring", "arguments": {"interval_ms": 100}})
        assert monitoring["result"]["structuredContent"]["started"]
        samples = call(8, "tools/call", {"name": "get_monitoring_results", "arguments": {}})
        assert samples["result"]["structuredContent"]["sample_count"] >= 1
        restarted = call(9, "tools/call", {"name": "restart_hub", "arguments": {}})
        assert restarted["result"]["structuredContent"]["run_dir"] == launched["result"]["structuredContent"]["run_dir"]
        campaign = call(10, "tools/call", {"name": "start_exploration", "arguments": {"steps": 1, "seed": 7}})
        campaign_id = campaign["result"]["structuredContent"]["id"]
        for _ in range(50):
            progress = call(11, "tools/call", {"name": "get_exploration", "arguments": {"id": campaign_id}})
            status = progress["result"]["structuredContent"]["status"]
            if status != "running":
                break
            time.sleep(0.1)
        assert status == "complete", progress
        assert progress["result"]["structuredContent"]["result"]["visited"] == 1
        fault = call(12, "tools/call", {"name": "inject_fault", "arguments": {"kind": "hub_termination"}})
        assert fault["result"]["structuredContent"]["injected"]
        assert Path(fault["result"]["structuredContent"]["evidence_path"]).is_file()
        violated = call(13, "tools/call", {"name": "evaluate_invariants", "arguments": {}})
        assert any(item["name"] == "hub_alive" for item in violated["result"]["structuredContent"]["violations"])
        defects = call(16, "tools/call", {"name": "list_detected_defects", "arguments": {}})
        assert defects["result"]["structuredContent"]["defects"] == []
        recovered = call(14, "tools/call", {"name": "restart_hub", "arguments": {}})
        assert recovered["result"]["structuredContent"]["run_dir"] == launched["result"]["structuredContent"]["run_dir"]
        stopped = call(15, "tools/call", {"name": "stop_hub", "arguments": {}})
        assert stopped["result"]["structuredContent"]["stopped"]
    finally:
        server.stdin.close()
        server.wait(timeout=10)
