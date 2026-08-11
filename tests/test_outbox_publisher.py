"""Unit test for OutboxPublisher claim/publish marking."""

from __future__ import annotations

from typing import Any

from kraken_server.outbox import ConnectionHub, OutboxPublisher


class _Result:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.rowcount = len(rows)

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _Connection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, dict[str, Any]]] = []

    def execute(self, statement: Any, params: dict[str, Any] | None = None):
        sql = str(statement)
        self.executed.append((sql, dict(params or {})))
        if "SELECT" in sql.upper():
            return _Result(self.rows)
        return _Result([])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Engine:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.connection = _Connection(rows)

    def begin(self):
        return self.connection


def test_outbox_publisher_marks_rows_published_and_notifies_hub() -> None:
    import pytest

    pytest.importorskip("sqlalchemy")
    hub = ConnectionHub()
    published: list[dict[str, Any]] = []
    hub.publish = published.append  # type: ignore[method-assign]
    rows = [
        {
            "outbox_id": 1,
            "event_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "project_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "event_type": "ProjectCreated",
            "payload": {},
            "position": 10,
            "revision": 0,
            "stream_id": "project:aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        }
    ]
    engine = _Engine(rows)
    publisher = OutboxPublisher(engine, hub, interval_seconds=60)
    count = publisher.publish_once()
    assert count == 1
    assert published[0]["type"] == "project_event"
    assert published[0]["position"] == 10
    assert published[0]["entity_kind"] == "project"
    assert published[0]["entity_id"] == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert any("UPDATE TRANSACTIONAL_OUTBOX" in sql.upper() for sql, _ in engine.connection.executed)
    assert published[1]["type"] == "catalog_changed"
