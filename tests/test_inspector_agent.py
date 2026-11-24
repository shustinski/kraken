"""Inspector owns a real local Agent and checks its authenticated health."""

from __future__ import annotations

from pathlib import Path

from kraken_inspector.session import InspectorSession


def test_inspector_launches_and_stops_agent() -> None:
    session = InspectorSession(Path(__file__).resolve().parents[1])
    try:
        launched = session.start_agent()
        assert launched["health"] == {"status": "ok", "api_version": "v1"}
        assert launched["pid"] > 0
        assert any(
            check["name"] == "agent_healthy" and check["status"] == "pass"
            for check in session.evaluate_invariants()["checks"]
        )
    finally:
        assert session.stop_agent()["stopped"]
