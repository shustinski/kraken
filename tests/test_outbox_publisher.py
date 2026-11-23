"""Unit test for OutboxPublisher claim/publish marking."""

from __future__ import annotations

from typing import Any

from kraken_server.outbox import ConnectionHub, OutboxPublisher, enqueue_notice


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


def test_notice_without_domain_event_preserves_entity_address() -> None:
    import pytest

    pytest.importorskip("sqlalchemy")
    connection = _Connection([])
    enqueue_notice(connection, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "job.state_changed", "job", "job-1")
    statement, params = connection.executed[0]
    assert "transactional_outbox" in statement
    assert params["payload"] == {"entity_kind": "job", "entity_id": "job-1"}
    assert params["event_type"] == "job.state_changed"


def test_publisher_uses_notice_payload_when_domain_event_is_absent() -> None:
    import pytest

    pytest.importorskip("sqlalchemy")
    hub = ConnectionHub()
    published: list[dict[str, Any]] = []
    hub.publish = published.append  # type: ignore[method-assign]
    engine = _Engine([{
        "outbox_id": 2,
        "event_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "project_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "event_type": "job.state_changed",
        "payload": {"entity_kind": "job", "entity_id": "job-1"},
        "position": None,
        "revision": None,
        "stream_id": None,
    }])
    assert OutboxPublisher(engine, hub).publish_once() == 1
    assert published == [{
        "type": "project_event",
        "project_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "event_type": "job.state_changed",
        "event_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "position": None, "revision": None, "stream_id": "",
        "entity_kind": "job", "entity_id": "job-1",
    }]


def test_notice_rolls_back_with_its_data_transaction() -> None:
    import pytest
    import sqlalchemy as sa

    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(sa.text("""CREATE TABLE transactional_outbox (
            event_id TEXT, project_id TEXT, event_type TEXT, payload TEXT, created_at TEXT
        )"""))
    with pytest.raises(RuntimeError, match="abort"):
        with engine.begin() as connection:
            enqueue_notice(connection, "project", "job.state_changed", "job", "job-1")
            raise RuntimeError("abort")
    with engine.connect() as connection:
        assert connection.execute(sa.text("SELECT COUNT(*) FROM transactional_outbox")).scalar_one() == 0


def test_publisher_emits_directory_envelope_for_directory_notice() -> None:
    hub = ConnectionHub()
    published: list[dict[str, Any]] = []
    hub.publish = published.append  # type: ignore[method-assign]
    engine = _Engine([{
        "outbox_id": 3, "event_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "project_id": "00000000-0000-0000-0000-000000000000",
        "event_type": "directory.performer_changed",
        "payload": {"entity_kind": "performer", "entity_id": "person"},
        "position": None, "revision": None, "stream_id": None,
    }])
    assert OutboxPublisher(engine, hub).publish_once() == 1
    assert published == [{"type": "directory_changed", "event_type": "directory.performer_changed",
                          "entity_kind": "performer", "entity_id": "person"}]
