"""SQLite identity, performer and project-ACL adapter for local Desktop."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from kraken_manager.domain.common import PrincipalId, ProjectId
from kraken_manager.domain.identity import (
    Principal,
    PrincipalProvider,
    ProjectRole,
    ProjectRoleAssignment,
    SystemRole,
)


class LocalIdentityAclStore:
    """Independent workstation identity/ACL catalog; no password bytes live here."""

    def __init__(self, database: Path | str) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS principals (
                    principal_id TEXT PRIMARY KEY,
                    external_key TEXT NOT NULL UNIQUE,
                    provider TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    issuer TEXT,
                    display_name TEXT NOT NULL,
                    email TEXT,
                    active INTEGER NOT NULL,
                    system_roles_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS project_acl (
                    project_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    assigned_by TEXT NOT NULL,
                    assigned_at TEXT NOT NULL,
                    revoked_at TEXT,
                    PRIMARY KEY(project_id, principal_id, role)
                );
                CREATE INDEX IF NOT EXISTS ix_project_acl_principal
                    ON project_acl(principal_id, project_id, revoked_at);
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _principal(row: sqlite3.Row) -> Principal:
        return Principal(
            id=PrincipalId(str(row["principal_id"])),
            provider=PrincipalProvider(str(row["provider"])),
            subject=str(row["subject"]),
            issuer=row["issuer"],
            display_name=str(row["display_name"]),
            email=row["email"],
            active=bool(row["active"]),
            system_roles=frozenset(SystemRole(value) for value in json.loads(row["system_roles_json"])),
        )

    def get(self, principal_id: PrincipalId) -> Principal | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM principals WHERE principal_id=?", (str(principal_id),)
            ).fetchone()
        return None if row is None else self._principal(row)

    def get_by_external_key(self, external_key: str) -> Principal | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM principals WHERE external_key=?", (external_key,)
            ).fetchone()
        return None if row is None else self._principal(row)

    def save(self, principal: Principal) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO principals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(principal_id) DO UPDATE SET
                    external_key=excluded.external_key,
                    provider=excluded.provider,
                    subject=excluded.subject,
                    issuer=excluded.issuer,
                    display_name=excluded.display_name,
                    email=excluded.email,
                    active=excluded.active,
                    system_roles_json=excluded.system_roles_json
                """,
                (
                    str(principal.id),
                    principal.external_key,
                    principal.provider.value,
                    principal.subject,
                    principal.issuer,
                    principal.display_name,
                    principal.email,
                    int(principal.active),
                    json.dumps(sorted(role.value for role in principal.system_roles)),
                ),
            )

    def roles_for(self, project_id: ProjectId, principal_id: PrincipalId) -> frozenset[ProjectRole]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role FROM project_acl WHERE project_id=? AND principal_id=? AND revoked_at IS NULL",
                (str(project_id), str(principal_id)),
            ).fetchall()
        return frozenset(ProjectRole(str(row["role"])) for row in rows)

    def assign(self, assignment: ProjectRoleAssignment) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO project_acl VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, principal_id, role) DO UPDATE SET
                    assigned_by=excluded.assigned_by,
                    assigned_at=excluded.assigned_at,
                    revoked_at=NULL
                """,
                (
                    str(assignment.project_id),
                    str(assignment.principal_id),
                    assignment.role.value,
                    str(assignment.assigned_by),
                    assignment.assigned_at.isoformat(),
                    None if assignment.revoked_at is None else assignment.revoked_at.isoformat(),
                ),
            )

    def revoke(self, project_id: ProjectId, principal_id: PrincipalId, role: ProjectRole) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE project_acl SET revoked_at=? WHERE project_id=? AND principal_id=? AND role=? AND revoked_at IS NULL",
                (datetime.now(UTC).isoformat(), str(project_id), str(principal_id), role.value),
            )


__all__ = ["LocalIdentityAclStore"]
