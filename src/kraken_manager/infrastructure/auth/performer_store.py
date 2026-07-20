"""SQLite performer catalog for the local Desktop profile."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.domain.common import PerformerId, PrincipalId
from kraken_manager.domain.identity import Performer

from .identity_store import LocalIdentityAclStore


class LocalSQLitePerformerStore:
    """Persist manual and principal-linked performers in a workstation DB."""

    def __init__(self, database: Path | str) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        # The optional principal link is a real foreign key.  Initializing the
        # colocated identity catalog first keeps this adapter useful on a fresh
        # workstation database as well as in the normal composition root.
        LocalIdentityAclStore(self.database)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS performers (
                    performer_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    color TEXT NOT NULL,
                    principal_id TEXT UNIQUE,
                    active INTEGER NOT NULL,
                    FOREIGN KEY(principal_id) REFERENCES principals(principal_id) ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS ix_performers_active_name
                    ON performers(active, name, performer_id);
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @staticmethod
    def _performer(row: sqlite3.Row) -> Performer:
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
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM performers WHERE performer_id=?", (str(performer_id),)
            ).fetchone()
        return None if row is None else self._performer(row)

    def get_by_principal(self, principal_id: PrincipalId) -> Performer | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM performers WHERE principal_id=?", (str(principal_id),)
            ).fetchone()
        return None if row is None else self._performer(row)

    def list(self, *, include_archived: bool = False) -> tuple[Performer, ...]:
        predicate = "" if include_archived else "WHERE active=1"
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM performers {predicate} "
                "ORDER BY name COLLATE NOCASE, performer_id"
            ).fetchall()
        performers = (self._performer(row) for row in rows)
        return tuple(sorted(performers, key=lambda item: (item.name.casefold(), str(item.id))))

    def create(self, performer: Performer) -> Performer:
        try:
            with self._write() as connection:
                connection.execute(
                    """
                    INSERT INTO performers(performer_id, name, color, principal_id, active)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(performer.id),
                        performer.name,
                        performer.color,
                        None if performer.principal_id is None else str(performer.principal_id),
                        int(performer.active),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ConflictError("performer id or principal link already exists") from exc
        return performer

    def update(self, performer: Performer) -> Performer:
        try:
            with self._write() as connection:
                row = connection.execute(
                    "SELECT * FROM performers WHERE performer_id=?", (str(performer.id),)
                ).fetchone()
                if row is None:
                    raise NotFoundError(f"Performer {performer.id} was not found")
                existing = self._performer(row)
                if existing.principal_id is not None and performer.principal_id != existing.principal_id:
                    raise ConflictError("an existing principal link cannot be rebound or removed")
                connection.execute(
                    """
                    UPDATE performers
                    SET name=?, color=?, principal_id=?, active=?
                    WHERE performer_id=?
                    """,
                    (
                        performer.name,
                        performer.color,
                        None if performer.principal_id is None else str(performer.principal_id),
                        int(performer.active),
                        str(performer.id),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ConflictError("principal is already linked to another performer") from exc
        return performer

    def archive(self, performer_id: PerformerId) -> Performer:
        with self._write() as connection:
            row = connection.execute(
                "SELECT * FROM performers WHERE performer_id=?", (str(performer_id),)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Performer {performer_id} was not found")
            performer = self._performer(row).archive()
            connection.execute(
                "UPDATE performers SET active=0 WHERE performer_id=?", (str(performer_id),)
            )
        return performer


__all__ = ["LocalSQLitePerformerStore"]
