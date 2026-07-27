"""Transactional PostgreSQL unit of work.

The connection is owned by the unit of work and is injected into every
metadata adapter.  This is the composition boundary which makes events,
current/temporal projections, ACL changes and the transactional outbox one
database transaction.  Blob writes are intentionally outside that
transaction: blobs are immutable and content addressed, so a rollback can at
worst leave an unreferenced object which an integrity/garbage-collection job
may remove later.
"""

from __future__ import annotations

from typing import Any, Self

from .event_store import PostgresEventStore
from .identity_store import PostgresIdentityAclStore
from .projection_store import PostgresProjectionStore


class PostgresUnitOfWork:
    def __init__(self, engine: Any, blobs: Any) -> None:
        self.engine = engine
        self.blobs = blobs
        self._connection: Any | None = None
        self._transaction: Any | None = None
        self._entered = False
        self._committed = False

        # Attributes are replaced with connection-bound adapters in __enter__.
        # Keeping them available before entry makes protocol/introspection
        # errors deterministic while still refusing accidental I/O.
        self.event_store: Any = None
        self.projections: Any = None
        self.identities: Any = None
        self.acl: Any = None

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("unit of work cannot be entered twice")
        self._connection = self.engine.connect()
        try:
            self._transaction = self._connection.begin()
            self.event_store = PostgresEventStore(self.engine, connection=self._connection)
            self.projections = PostgresProjectionStore(self.engine, connection=self._connection)
            identity_acl = PostgresIdentityAclStore(self.engine, connection=self._connection)
            self.identities = identity_acl
            self.acl = identity_acl
            self._entered = True
            self._committed = False
            return self
        except BaseException:
            self._connection.close()
            self._connection = None
            self._transaction = None
            raise

    def commit(self) -> None:
        if not self._entered or self._transaction is None:
            raise RuntimeError("unit of work is not active")
        if self._committed:
            raise RuntimeError("unit of work was already committed")
        self._transaction.commit()
        self._committed = True

    def rollback(self) -> None:
        if self._transaction is not None and self._transaction.is_active:
            self._transaction.rollback()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        try:
            if exc_type is not None or not self._committed:
                self.rollback()
        finally:
            if self._connection is not None:
                self._connection.close()
            self._connection = None
            self._transaction = None
            self._entered = False
        return False


class PostgresUnitOfWorkFactory:
    """Create transaction-scoped metadata adapters over one engine."""

    def __init__(self, engine: Any, blobs: Any) -> None:
        self.engine = engine
        self.blobs = blobs

    def __call__(self) -> PostgresUnitOfWork:
        return PostgresUnitOfWork(self.engine, self.blobs)


__all__ = ["PostgresUnitOfWork", "PostgresUnitOfWorkFactory"]
