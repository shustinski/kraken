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
    hub.register(project_ws)
    hub.register(catalog_ws)
    hub.register(other_ws)
    hub.subscribe(project_ws, project_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    hub.subscribe(catalog_ws, catalog=True)

    async def _run() -> None:
        await hub._fanout(
            {
                "type": "project_event",
                "project_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "event_type": "ProjectCreated",
                "event_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "position": 1,
                "revision": 0,
            }
        )

    asyncio.run(_run())

    assert project_ws.messages
    assert catalog_ws.messages
    assert not other_ws.messages
    assert project_ws.messages[0]["event_type"] == "ProjectCreated"


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
