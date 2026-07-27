"""Hashed cache for federated sessions used by read-only outage fallback."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Iterator

from kraken_manager.domain.common import PrincipalId

from .event_store import _sqlalchemy
from .identity_store import _identity_tables


def _session_table() -> tuple[Any, Any]:
    sa, _ = _sqlalchemy()
    metadata, principals, _ = _identity_tables()
    table = sa.Table(
        "federated_sessions",
        metadata,
        sa.Column("token_hash", sa.Text, primary_key=True),
        sa.Column(
            "principal_id",
            sa.Uuid(as_uuid=False),
            sa.ForeignKey(principals.c.principal_id, ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.Text, nullable=False),
        sa.Column("last_verified_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    return metadata, table


class PostgresFederatedSessionCache:
    """Never stores bearer bytes; cached entries authorize reads only."""

    def __init__(self, engine: Any, *, lifetime: timedelta = timedelta(hours=12)) -> None:
        if lifetime <= timedelta(0):
            raise ValueError("session cache lifetime must be positive")
        self.engine = engine
        self.lifetime = lifetime
        self.metadata, self.table = _session_table()

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def save(self, token: str, principal_id: PrincipalId, *, provider: str = "gitlab") -> None:
        _, pg_insert = _sqlalchemy()
        now = datetime.now(UTC)
        values = {
            "token_hash": self._hash(token),
            "principal_id": str(principal_id),
            "provider": provider,
            "last_verified_at": now,
            "expires_at": now + self.lifetime,
        }
        with self.engine.begin() as connection:
            connection.execute(
                pg_insert(self.table)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=[self.table.c.token_hash],
                    set_={key: value for key, value in values.items() if key != "token_hash"},
                )
            )

    def resolve(self, token: str) -> tuple[PrincipalId, str] | None:
        sa, _ = _sqlalchemy()
        now = datetime.now(UTC)
        with self.engine.begin() as connection:
            row = connection.execute(
                sa.select(self.table.c.principal_id, self.table.c.provider).where(
                    self.table.c.token_hash == self._hash(token),
                    self.table.c.expires_at > now,
                )
            ).first()
            connection.execute(sa.delete(self.table).where(self.table.c.expires_at <= now))
        if row is None:
            return None
        return PrincipalId(str(row.principal_id)), str(row.provider)

    def revoke(self, token: str) -> None:
        sa, _ = _sqlalchemy()
        with self.engine.begin() as connection:
            connection.execute(sa.delete(self.table).where(self.table.c.token_hash == self._hash(token)))


__all__ = ["PostgresFederatedSessionCache"]
