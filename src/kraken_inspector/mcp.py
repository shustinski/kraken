"""Small standards-compatible MCP stdio transport for the external Inspector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import BinaryIO

from .session import InspectorSession

PROTOCOL_VERSION = "2025-06-18"

TOOLS = [
    (
        "launch_hub",
        "Start an isolated real Kraken Hub",
        {"timeout": {"type": "number"}},
    ),
    (
        "start_server",
        "Start disposable PostgreSQL and Kraken Server",
        {"postgres_bin": {"type": "string"}, "timeout": {"type": "number"}},
    ),
    ("stop_server", "Stop the owned Server and PostgreSQL cluster", {}),
    ("start_agent", "Start an isolated local Kraken Agent control service", {"timeout": {"type": "number"}}),
    ("stop_agent", "Stop the owned local Agent", {}),
    ("inspect_environment", "Inspect revision, process and resource state", {}),
    ("inspect_ui", "List visible windows and Qt elements", {}),
    (
        "wait_for_ui",
        "Wait for a visible window title or element text",
        {"window_title": {"type": "string"}, "text": {"type": "string"}, "timeout": {"type": "number"}},
    ),
    (
        "perform_action",
        "Send a real UI event to a discovered element",
        {
            "id": {"type": "string"},
            "action": {"type": "string"},
            "value": {"type": "string"},
            "x": {"type": "number"},
            "y": {"type": "number"},
        },
    ),
    ("capture_screenshot", "Save a window screenshot", {"window": {"type": "integer"}}),
    ("get_logs", "Get a bounded Hub log tail", {"lines": {"type": "integer"}}),
    (
        "start_monitoring",
        "Sample process resources until stopped",
        {"interval_ms": {"type": "integer"}, "max_rss_growth_bytes": {"type": "integer"}},
    ),
    ("get_monitoring_results", "Read bounded resource samples and policy anomalies", {"limit": {"type": "integer"}}),
    ("stop_monitoring", "Stop resource sampling", {}),
    ("evaluate_invariants", "Check process liveness and UI responsiveness", {}),
    ("save_scenario", "Save recorded UI actions", {}),
    ("replay_scenario", "Replay a saved Inspector scenario", {"path": {"type": "string"}}),
    (
        "explore_states",
        "Explore enabled UI actions with a reproducible seed",
        {"steps": {"type": "integer"}, "seed": {"type": "integer"}},
    ),
    (
        "start_exploration",
        "Start a bounded background exploration campaign",
        {"steps": {"type": "integer"}, "seed": {"type": "integer"}},
    ),
    ("get_exploration", "Get campaign progress and results", {"id": {"type": "string"}}),
    ("cancel_exploration", "Cancel a running campaign", {"id": {"type": "string"}}),
    ("list_detected_defects", "List unexpected invariant violations with evidence", {}),
    ("inject_fault", "Kill the owned isolated Hub process; use restart_hub to recover", {"kind": {"type": "string"}}),
    ("stop_hub", "Stop only the Hub and descendants owned by this session", {}),
    ("restart_hub", "Restart Hub with the same isolated data and settings", {"timeout": {"type": "number"}}),
]


def _tool_spec(name: str, description: str, properties: dict) -> dict:
    required = {
        "perform_action": ["id", "action"],
        "start_server": ["postgres_bin"],
        "replay_scenario": ["path"],
        "get_exploration": ["id"],
        "cancel_exploration": ["id"],
        "inject_fault": ["kind"],
    }.get(name, [])
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


class InspectorMCP:
    def __init__(self, session: InspectorSession) -> None:
        self.session = session

    def dispatch(self, request: dict) -> dict | None:
        request_id = request.get("id")
        method = request.get("method")
        if request_id is None:
            return None
        if method == "initialize":
            return self._success(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "kraken-inspector", "version": "0.1.0"},
                },
            )
        if method == "ping":
            return self._success(request_id, {})
        if method == "tools/list":
            return self._success(request_id, {"tools": [_tool_spec(*item) for item in TOOLS]})
        if method != "tools/call":
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "Method not found"}}
        params = request.get("params", {})
        name = params.get("name", "")
        args = params.get("arguments", {})
        if name not in {item[0] for item in TOOLS} or not isinstance(args, dict):
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": "Unknown tool or invalid arguments"},
            }
        try:
            value = self._call(name, args)
            return self._success(
                request_id,
                {
                    "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}],
                    "structuredContent": value,
                },
            )
        except (ValueError, RuntimeError, ConnectionError, TimeoutError, OSError) as exc:
            return self._success(
                request_id, {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True}
            )

    @staticmethod
    def _success(request_id: object, value: dict) -> dict:
        return {"jsonrpc": "2.0", "id": request_id, "result": value}

    def _call(self, name: str, args: dict) -> dict:
        session = self.session
        if name == "launch_hub":
            return session.launch_hub(timeout=float(args.get("timeout", 45)))
        if name == "start_server":
            return session.start_server(postgres_bin=Path(args["postgres_bin"]), timeout=float(args.get("timeout", 60)))
        if name == "stop_server":
            return session.stop_server()
        if name == "start_agent":
            return session.start_agent(timeout=float(args.get("timeout", 15)))
        if name == "stop_agent":
            return session.stop_agent()
        if name == "inspect_environment":
            return session.environment()
        if name == "inspect_ui":
            return session.inspect_ui()
        if name == "wait_for_ui":
            return session.wait_for_ui(
                window_title=str(args.get("window_title", "")),
                text=str(args.get("text", "")),
                timeout=float(args.get("timeout", 10)),
            )
        if name == "perform_action":
            return session.perform_action(
                str(args["id"]),
                str(args["action"]),
                str(args.get("value", "")),
                x=float(args.get("x", 0.5)),
                y=float(args.get("y", 0.5)),
            )
        if name == "capture_screenshot":
            return session.screenshot(int(args.get("window", 0)))
        if name == "get_logs":
            return {"text": session.get_logs(int(args.get("lines", 100)))}
        if name == "start_monitoring":
            return session.start_monitoring(
                interval_ms=int(args.get("interval_ms", 500)), max_rss_growth_bytes=args.get("max_rss_growth_bytes")
            )
        if name == "get_monitoring_results":
            return session.get_monitoring_results(limit=int(args.get("limit", 100)))
        if name == "stop_monitoring":
            return session.stop_monitoring()
        if name == "evaluate_invariants":
            return session.evaluate_invariants()
        if name == "save_scenario":
            return {"path": str(session.save_scenario())}
        if name == "replay_scenario":
            return session.replay_scenario(Path(args["path"]))
        if name == "explore_states":
            return session.explore_states(steps=int(args.get("steps", 10)), seed=int(args.get("seed", 0)))
        if name == "start_exploration":
            return session.start_exploration(steps=int(args.get("steps", 10)), seed=int(args.get("seed", 0)))
        if name == "get_exploration":
            return session.get_exploration(str(args["id"]))
        if name == "cancel_exploration":
            return session.cancel_exploration(str(args["id"]))
        if name == "list_detected_defects":
            return {"defects": session.defects}
        if name == "inject_fault":
            return session.inject_fault(str(args["kind"]))
        if name == "stop_hub":
            return session.stop()
        if name == "restart_hub":
            return session.restart_hub(timeout=float(args.get("timeout", 45)))
        raise ValueError(f"Unknown tool: {name}")


def _read_message(stream: BinaryIO) -> dict | None:
    line = stream.readline()
    if not line:
        return None
    if len(line) > 4_000_000:
        raise ValueError("MCP message too large")
    return json.loads(line)


def _write_message(stream: BinaryIO, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    stream.write(body + b"\n")
    stream.flush()


def serve(server: InspectorMCP, input_stream: BinaryIO, output_stream: BinaryIO) -> None:
    try:
        while (request := _read_message(input_stream)) is not None:
            response = server.dispatch(request)
            if response is not None:
                _write_message(output_stream, response)
    finally:
        server.session.stop()
        server.session.stop_agent()
        server.session.stop_server()
