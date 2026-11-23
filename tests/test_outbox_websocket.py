"""Tests for outbox ConnectionHub and WebSocket subscribe protocol."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from kraken_server.outbox import ConnectionHub


class _FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.messages.append(payload)


def test_connection_hub_fans_out_to_project_and_catalog_subscribers() -> None:
    hub = ConnectionHub()
    project_ws = _FakeWebSocket()
    catalog_ws = _FakeWebSocket()
    other_ws = _FakeWebSocket()
    hidden_ws = _FakeWebSocket()
    hub.register(project_ws)
    hub.register(catalog_ws)
    hub.register(other_ws)
    hub.register(hidden_ws)
    project_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    hub.subscribe(project_ws, project_id=project_id)
    hub.subscribe(catalog_ws, catalog=True, catalog_visible=lambda candidate: candidate == project_id)
    hub.subscribe(hidden_ws, catalog=True, catalog_visible=lambda _candidate: False)

    async def _run() -> None:
        await hub._fanout(
            {
                "type": "project_event",
                "project_id": project_id,
                "event_type": "LayerCreated",
                "event_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "position": 1,
                "revision": 0,
                "stream_id": "layer:cccccccc-cccc-cccc-cccc-cccccccccccc",
            }
        )
        await hub._fanout(
            {
                "type": "catalog_changed",
                "project_id": project_id,
                "event_type": "ProjectCreated",
                "event_id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
            }
        )

    asyncio.run(_run())

    assert [message["type"] for message in project_ws.messages] == ["project_event", "catalog_changed"]
    assert [message["type"] for message in catalog_ws.messages] == ["catalog_changed"]
    assert "stream_id" not in catalog_ws.messages[0]
    assert not other_ws.messages
    assert not hidden_ws.messages


def test_directory_change_reaches_all_catalog_subscribers() -> None:
    hub = ConnectionHub()
    catalog_ws = _FakeWebSocket()
    hidden_ws = _FakeWebSocket()
    project_ws = _FakeWebSocket()
    hub.subscribe(catalog_ws, catalog=True, catalog_visible=lambda _project_id: True)
    hub.subscribe(hidden_ws, catalog=True, catalog_visible=lambda _project_id: False)
    hub.subscribe(project_ws, project_id="project")

    asyncio.run(hub._fanout({"type": "directory_changed", "entity_kind": "performer", "entity_id": "person"}))

    assert [item["type"] for item in catalog_ws.messages] == ["directory_changed"]
    assert [item["type"] for item in hidden_ws.messages] == ["directory_changed"]
    assert not project_ws.messages


def test_role_revocation_wakes_catalog_without_disclosing_project_id() -> None:
    hub = ConnectionHub()
    revoked = _FakeWebSocket()
    hub.subscribe(revoked, project_id="secret", catalog=True, catalog_visible=lambda _project_id: False)

    asyncio.run(hub._fanout({"type": "catalog_changed", "event_type": "ProjectRoleRevoked"}))

    assert revoked.messages == [{"type": "catalog_changed", "event_type": "ProjectRoleRevoked"}]
    asyncio.run(hub._fanout({"type": "project_event", "project_id": "secret",
                             "event_type": "ProjectRoleRevoked"}))
    asyncio.run(hub._fanout({"type": "project_event", "project_id": "secret",
                             "event_type": "LayerCreated"}))
    assert len(revoked.messages) == 1


def test_websocket_subscribe_protocol_with_fastapi() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kraken_server.app import create_app
    from kraken_server.outbox import ConnectionHub

    hub = ConnectionHub()
    app = create_app(development=True, connection_hub=hub)
    client = TestClient(app)
    with client.websocket_connect("/api/v1/ws", headers={"Authorization": "Bearer developer"}) as ws:
        connected = ws.receive_json()
        assert connected["type"] == "connected"
        ws.send_json({"type": "subscribe", "project_id": "p1", "catalog": True})
        subscribed = ws.receive_json()
        assert subscribed["type"] == "subscribed"
        assert subscribed["catalog"] is True
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"
