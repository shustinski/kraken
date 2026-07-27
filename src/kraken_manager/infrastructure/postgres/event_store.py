"""SQLAlchemy 2-style PostgreSQL event store with outbox and idempotency."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from kraken_manager.application.errors import ConcurrencyError
from kraken_manager.domain.common import PerformerId, PrincipalId, ProjectId
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
from kraken_manager.domain.identity import PrincipalProvider


class PostgresRevisionConflict(ConcurrencyError):
    pass


def _sqlalchemy() -> Any:
    try:
        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import insert as pg_insert
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install Kraken with the 'postgres' extra") from exc
    return sa, pg_insert


def _tables() -> tuple[Any, Any, Any, Any, Any]:
    sa, _ = _sqlalchemy()
    metadata = sa.MetaData()
    streams = sa.Table(
        "event_streams",
        metadata,
        sa.Column("stream_id", sa.Text, primary_key=True),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False, index=True),
        sa.Column("revision", sa.BigInteger, nullable=False, server_default="0"),
    )
    events = sa.Table(
        "domain_events",
        metadata,
        sa.Column("position", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column("event_id", sa.Uuid(as_uuid=False), nullable=False, unique=True),
        sa.Column("stream_id", sa.Text, nullable=False),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("revision", sa.BigInteger, nullable=False),
        sa.Column("event_type", sa.Text, nullable=False),
        sa.Column("schema_version", sa.Integer, nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_at", sa.DateTime(timezone=True)),
        sa.Column("actor", sa.JSON, nullable=False),
        sa.Column("performer_id", sa.Uuid(as_uuid=False)),
        sa.Column("program", sa.JSON),
        sa.Column("correlation_id", sa.Uuid(as_uuid=False)),
        sa.Column("causation_id", sa.Uuid(as_uuid=False)),
        sa.Column("idempotency_key", sa.Text),
        sa.Column("payload", sa.JSON, nullable=False),
        sa.UniqueConstraint("stream_id", "revision", name="uq_domain_events_stream_revision"),
    )
    sa.Index("ix_domain_events_project_time", events.c.project_id, events.c.recorded_at)
    sa.Index("ix_domain_events_project_type_time", events.c.project_id, events.c.event_type, events.c.recorded_at)
    sa.Index("ix_domain_events_project_idempotency", events.c.project_id, events.c.idempotency_key)
    command_keys = sa.Table(
        "command_idempotency",
        metadata,
        sa.Column("project_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("idempotency_key", sa.Text, primary_key=True),
        sa.Column("event_ids", sa.JSON, nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    )
    outbox = sa.Table(
        "transactional_outbox",
        metadata,
        sa.Column("outbox_id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column("event_id", sa.Uuid(as_uuid=False), nullable=False, unique=True),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("event_type", sa.Text, nullable=False),
        sa.Column("payload", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
    )
    return metadata, streams, events, command_keys, outbox


def _actor_dict(actor: ActorSnapshot) -> dict[str, Any]:
    return {
        "principal_id": str(actor.principal_id),
        "provider": actor.provider.value,
        "subject": actor.subject,
        "display_name": actor.display_name,
    }


def _program_dict(program: ProgramSnapshot | None) -> dict[str, Any] | None:
    return None if program is None else {"name": program.name, "version": program.version}


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _event_values(event: EventEnvelope) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "stream_id": event.stream_id,
        "project_id": str(event.project_id),
        "revision": event.revision,
        "event_type": event.event_type,
        "schema_version": event.schema_version,
        "recorded_at": event.recorded_at,
        "effective_at": event.effective_at,
        "actor": _actor_dict(event.actor),
        "performer_id": None if event.performer_id is None else str(event.performer_id),
        "program": _program_dict(event.program),
        "correlation_id": event.correlation_id,
        "causation_id": event.causation_id,
        "idempotency_key": event.idempotency_key,
        "payload": _plain(event.payload),
    }


def _event_from_row(row: Mapping[str, Any]) -> EventEnvelope:
    actor = row["actor"]
    program = row.get("program")
    return EventEnvelope(
        event_id=str(row["event_id"]),
        stream_id=str(row["stream_id"]),
        project_id=ProjectId(str(row["project_id"])),
        revision=int(row["revision"]),
        event_type=str(row["event_type"]),
        payload=dict(row["payload"]),
        schema_version=int(row["schema_version"]),
        recorded_at=row["recorded_at"],
        effective_at=row.get("effective_at"),
        actor=ActorSnapshot(
            principal_id=PrincipalId(str(actor["principal_id"])),
            provider=PrincipalProvider(str(actor["provider"])),
            subject=str(actor["subject"]),
            display_name=str(actor["display_name"]),
        ),
        performer_id=None if row.get("performer_id") is None else PerformerId(str(row["performer_id"])),
        program=None
        if program is None
        else ProgramSnapshot(name=str(program["name"]), version=program.get("version")),
        correlation_id=None if row.get("correlation_id") is None else str(row["correlation_id"]),
        causation_id=None if row.get("causation_id") is None else str(row["causation_id"]),
        idempotency_key=row.get("idempotency_key"),
    )


class PostgresEventStore:
    """Canonical event storage. Alembic, not this class, owns production DDL."""

    def __init__(self, engine: Any, *, connection: Any | None = None, create_schema_for_tests: bool = False) -> None:
        self.engine = engine
        self.connection = connection
        self.metadata, self.streams, self.events, self.command_keys, self.outbox = _tables()
        if create_schema_for_tests:
            self.metadata.create_all(engine)

    @contextmanager
    def _scope(self, *, write: bool = False) -> Iterator[Any]:
        if self.connection is not None:
            yield self.connection
            return
        context = self.engine.begin() if write else self.engine.connect()
        with context as connection:
            yield connection

    def load_stream(
        self,
        stream_id: str,
        *,
        after_revision: int = 0,
        as_of: datetime | None = None,
    ) -> tuple[EventEnvelope, ...]:
        sa, _ = _sqlalchemy()
        statement = sa.select(self.events).where(
            self.events.c.stream_id == stream_id,
            self.events.c.revision > after_revision,
        )
        if as_of is not None:
            statement = statement.where(self.events.c.recorded_at <= as_of)
        statement = statement.order_by(self.events.c.revision)
        with self._scope() as connection:
            rows = connection.execute(statement).mappings().all()
        return tuple(_event_from_row(row) for row in rows)

    def current_revision(self, stream_id: str) -> int:
        sa, _ = _sqlalchemy()
        with self._scope() as connection:
            value = connection.execute(
                sa.select(self.streams.c.revision).where(self.streams.c.stream_id == stream_id)
            ).scalar_one_or_none()
        return 0 if value is None else int(value)

    def append(
        self,
        stream_id: str,
        *,
        expected_revision: int,
        events: Sequence[EventEnvelope],
    ) -> int:
        if expected_revision < 0:
            raise ValueError("expected_revision cannot be negative")
        if not events:
            actual = self.current_revision(stream_id)
            if actual != expected_revision:
                raise PostgresRevisionConflict(f"Expected {expected_revision}, found {actual}")
            return actual
        project_ids = {str(event.project_id) for event in events}
        if len(project_ids) != 1 or any(event.stream_id != stream_id for event in events):
            raise ValueError("All events must belong to the supplied stream and one project")
        expected_event_revisions = list(range(expected_revision + 1, expected_revision + len(events) + 1))
        if [event.revision for event in events] != expected_event_revisions:
            raise ValueError("Event envelope revisions do not continue expected_revision")
        project_id = next(iter(project_ids))
        sa, pg_insert = _sqlalchemy()
        with self._scope(write=True) as connection:
            connection.execute(
                pg_insert(self.streams)
                .values(stream_id=stream_id, project_id=project_id, revision=0)
                .on_conflict_do_nothing(index_elements=[self.streams.c.stream_id])
            )
            stream_row = connection.execute(
                sa.select(self.streams).where(self.streams.c.stream_id == stream_id).with_for_update()
            ).mappings().one()
            actual = int(stream_row["revision"])
            if actual != expected_revision:
                raise PostgresRevisionConflict(f"Expected {expected_revision}, found {actual}")

            keys = {event.idempotency_key for event in events if event.idempotency_key is not None}
            if len(keys) > 1:
                raise ValueError("One append cannot contain multiple idempotency keys")
            if keys:
                key = next(iter(keys))
                inserted = connection.execute(
                    pg_insert(self.command_keys)
                    .values(
                        project_id=project_id,
                        idempotency_key=key,
                        event_ids=[event.event_id for event in events],
                        recorded_at=events[0].recorded_at,
                    )
                    .on_conflict_do_nothing(
                        index_elements=[self.command_keys.c.project_id, self.command_keys.c.idempotency_key]
                    )
                )
                if inserted.rowcount != 1:
                    raise PostgresRevisionConflict("Idempotency key was already committed")
            values = [_event_values(event) for event in events]
            connection.execute(sa.insert(self.events), values)
            connection.execute(
                sa.insert(self.outbox),
                [
                    {
                        "event_id": event.event_id,
                        "project_id": str(event.project_id),
                        "event_type": event.event_type,
                        "payload": _plain(event.payload),
                        "created_at": event.recorded_at,
                    }
                    for event in events
                ],
            )
            connection.execute(
                sa.update(self.streams)
                .where(self.streams.c.stream_id == stream_id)
                .values(revision=expected_revision + len(events))
            )
        return expected_revision + len(events)

    def find_by_idempotency_key(
        self, project_id: ProjectId, idempotency_key: str
    ) -> tuple[EventEnvelope, ...]:
        sa, _ = _sqlalchemy()
        statement = (
            sa.select(self.events)
            .where(
                self.events.c.project_id == str(project_id),
                self.events.c.idempotency_key == idempotency_key,
            )
            .order_by(self.events.c.position)
        )
        with self._scope() as connection:
            rows = connection.execute(statement).mappings().all()
        return tuple(_event_from_row(row) for row in rows)

    def list_project_events(
        self,
        project_id: ProjectId,
        *,
        after_position: int = 0,
        limit: int = 100,
        as_of: datetime | None = None,
    ) -> tuple[tuple[int, EventEnvelope], ...]:
        """Cursor-friendly history across every stream of one project."""
        if after_position < 0:
            raise ValueError("after_position cannot be negative")
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        sa, _ = _sqlalchemy()
        statement = sa.select(self.events).where(
            self.events.c.project_id == str(project_id),
            self.events.c.position > after_position,
        )
        if as_of is not None:
            statement = statement.where(self.events.c.recorded_at <= as_of)
        statement = statement.order_by(self.events.c.position).limit(limit)
        with self._scope() as connection:
            rows = connection.execute(statement).mappings().all()
        return tuple((int(row["position"]), _event_from_row(row)) for row in rows)


__all__ = ["PostgresEventStore", "PostgresRevisionConflict"]
