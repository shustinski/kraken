"""PostgreSQL performer catalog adapter."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.domain.common import PerformerId, PrincipalId
from kraken_manager.domain.identity import Performer

from .event_store import _sqlalchemy
from .identity_store import _identity_tables


def _performer_tables() -> tuple[Any, Any]:
    sa, _ = _sqlalchemy()
    metadata, principals, _ = _identity_tables()
    performers = sa.Table(
        "performers",
        metadata,
        sa.Column("performer_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column(
            "principal_id",
            sa.Uuid(as_uuid=False),
            sa.ForeignKey(principals.c.principal_id, ondelete="RESTRICT"),
            unique=True,
        ),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("color", sa.String(7), nullable=False),
        sa.Column("active", sa.Boolean, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    sa.Index("ix_performers_active_name", performers.c.active, performers.c.name, performers.c.performer_id)
    return metadata, performers


class PostgresPerformerStore:
    """Multi-writer performer store with a unique optional principal link."""

    def __init__(
        self,
        engine: Any,
        *,
        connection: Any | None = None,
        create_schema_for_tests: bool = False,
    ) -> None:
        self.engine = engine
        self.connection = connection
        self.metadata, self.performers = _performer_tables()
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

    @staticmethod
    def _performer(row: Any) -> Performer:
        return Performer(
            id=PerformerId(str(row["performer_id"])),
            name=str(row["name"]),
            color=str(row["color"]),
            principal_id=(
                None if row["principal_id"] is None else PrincipalId(str(row["principal_id"]))
            ),
            active=bool(row["active"]),
        )

    def get(self, performer_id: PerformerId) -> Performer | None:
        sa, _ = _sqlalchemy()
        with self._scope() as connection:
            row = connection.execute(
                sa.select(self.performers).where(self.performers.c.performer_id == str(performer_id))
            ).mappings().first()
        return None if row is None else self._performer(row)

    def get_by_principal(self, principal_id: PrincipalId) -> Performer | None:
        sa, _ = _sqlalchemy()
        with self._scope() as connection:
            row = connection.execute(
                sa.select(self.performers).where(self.performers.c.principal_id == str(principal_id))
            ).mappings().first()
        return None if row is None else self._performer(row)

    def list(self, *, include_archived: bool = False) -> tuple[Performer, ...]:
        sa, _ = _sqlalchemy()
        query = sa.select(self.performers)
        if not include_archived:
            query = query.where(self.performers.c.active.is_(True))
        query = query.order_by(self.performers.c.name, self.performers.c.performer_id)
        with self._scope() as connection:
            rows = connection.execute(query).mappings().all()
        performers = (self._performer(row) for row in rows)
        return tuple(sorted(performers, key=lambda item: (item.name.casefold(), str(item.id))))

    def create(self, performer: Performer) -> Performer:
        _, pg_insert = _sqlalchemy()
        now = datetime.now(UTC)
        statement = (
            pg_insert(self.performers)
            .values(
                performer_id=str(performer.id),
                principal_id=(
                    None if performer.principal_id is None else str(performer.principal_id)
                ),
                name=performer.name,
                color=performer.color,
                active=performer.active,
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_nothing()
            .returning(self.performers.c.performer_id)
        )
        with self._scope(write=True) as connection:
            inserted_id = connection.execute(statement).scalar_one_or_none()
            if inserted_id is None:
                raise ConflictError("performer id or principal link already exists")
        return performer

    def update(self, performer: Performer) -> Performer:
        sa, _ = _sqlalchemy()
        with self._scope(write=True) as connection:
            current_row = connection.execute(
                sa.select(self.performers).where(self.performers.c.performer_id == str(performer.id))
            ).mappings().first()
            if current_row is None:
                raise NotFoundError(f"Performer {performer.id} was not found")
            current = self._performer(current_row)
            if current.principal_id is not None and performer.principal_id != current.principal_id:
                raise ConflictError("an existing principal link cannot be rebound or removed")
            if performer.principal_id is not None:
                conflict = connection.execute(
                    sa.select(self.performers.c.performer_id).where(
                        self.performers.c.principal_id == str(performer.principal_id),
                        self.performers.c.performer_id != str(performer.id),
                    )
                ).first()
                if conflict is not None:
                    raise ConflictError("principal is already linked to another performer")
            try:
                connection.execute(
                    sa.update(self.performers)
                    .where(self.performers.c.performer_id == str(performer.id))
                    .values(
                        principal_id=(
                            None if performer.principal_id is None else str(performer.principal_id)
                        ),
                        name=performer.name,
                        color=performer.color,
                        active=performer.active,
                        updated_at=datetime.now(UTC),
                    )
                )
            except sa.exc.IntegrityError as exc:
                raise ConflictError("principal is already linked to another performer") from exc
        return performer

    def archive(self, performer_id: PerformerId) -> Performer:
        sa, _ = _sqlalchemy()
        statement = (
            sa.update(self.performers)
            .where(self.performers.c.performer_id == str(performer_id))
            .values(active=False, updated_at=datetime.now(UTC))
            .returning(*self.performers.c)
        )
        with self._scope(write=True) as connection:
            row = connection.execute(statement).mappings().first()
            if row is None:
                raise NotFoundError(f"Performer {performer_id} was not found")
        return self._performer(row)


__all__ = ["PostgresPerformerStore"]
