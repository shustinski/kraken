"""PostgreSQL Principal and Kraken-owned project ACL adapters."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Iterator

from kraken_manager.domain.common import PrincipalId, ProjectId
from kraken_manager.domain.identity import (
    Principal,
    PrincipalProvider,
    ProjectRole,
    ProjectRoleAssignment,
    SystemRole,
)

from .event_store import _sqlalchemy


def _identity_tables() -> tuple[Any, Any, Any]:
    sa, _ = _sqlalchemy()
    metadata = sa.MetaData()
    principals = sa.Table(
        "principals",
        metadata,
        sa.Column("principal_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("external_key", sa.Text, nullable=False, unique=True),
        sa.Column("provider", sa.Text, nullable=False),
        sa.Column("issuer", sa.Text),
        sa.Column("subject", sa.Text, nullable=False),
        sa.Column("display_name", sa.Text, nullable=False),
        sa.Column("email", sa.Text),
        sa.Column("enabled", sa.Boolean, nullable=False),
        sa.Column("system_roles", sa.JSON, nullable=False),
    )
    acl = sa.Table(
        "project_acl",
        metadata,
        sa.Column("project_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("principal_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("role", sa.Text, primary_key=True),
        sa.Column("granted_by", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("granted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
    )
    return metadata, principals, acl


class PostgresIdentityAclStore:
    def __init__(self, engine: Any, *, connection: Any | None = None, create_schema_for_tests: bool = False) -> None:
        self.engine = engine
        self.connection = connection
        self.metadata, self.principals, self.acl = _identity_tables()
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
    def _principal(row: Any) -> Principal:
        return Principal(
            id=PrincipalId(str(row["principal_id"])),
            provider=PrincipalProvider(str(row["provider"])),
            subject=str(row["subject"]),
            issuer=row["issuer"],
            display_name=str(row["display_name"]),
            email=row["email"],
            active=bool(row["enabled"]),
            system_roles=frozenset(SystemRole(value) for value in row["system_roles"]),
        )

    def get(self, principal_id: PrincipalId) -> Principal | None:
        sa, _ = _sqlalchemy()
        with self._scope() as connection:
            row = connection.execute(
                sa.select(self.principals).where(self.principals.c.principal_id == str(principal_id))
            ).mappings().first()
        return None if row is None else self._principal(row)

    def get_by_external_key(self, external_key: str) -> Principal | None:
        sa, _ = _sqlalchemy()
        with self._scope() as connection:
            row = connection.execute(
                sa.select(self.principals).where(self.principals.c.external_key == external_key)
            ).mappings().first()
        return None if row is None else self._principal(row)

    def save(self, principal: Principal) -> None:
        _, pg_insert = _sqlalchemy()
        values = {
            "principal_id": str(principal.id),
            "external_key": principal.external_key,
            "provider": principal.provider.value,
            "issuer": principal.issuer,
            "subject": principal.subject,
            "display_name": principal.display_name,
            "email": principal.email,
            "enabled": principal.active,
            "system_roles": sorted(role.value for role in principal.system_roles),
        }
        with self._scope(write=True) as connection:
            connection.execute(
                pg_insert(self.principals)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=[self.principals.c.principal_id],
                    set_={key: value for key, value in values.items() if key != "principal_id"},
                )
            )

    def roles_for(self, project_id: ProjectId, principal_id: PrincipalId) -> frozenset[ProjectRole]:
        sa, _ = _sqlalchemy()
        with self._scope() as connection:
            roles = connection.execute(
                sa.select(self.acl.c.role).where(
                    self.acl.c.project_id == str(project_id),
                    self.acl.c.principal_id == str(principal_id),
                    self.acl.c.revoked_at.is_(None),
                )
            ).scalars().all()
        return frozenset(ProjectRole(str(role)) for role in roles)

    def assign(self, assignment: ProjectRoleAssignment) -> None:
        _, pg_insert = _sqlalchemy()
        values = {
            "project_id": str(assignment.project_id),
            "principal_id": str(assignment.principal_id),
            "role": assignment.role.value,
            "granted_by": str(assignment.assigned_by),
            "granted_at": assignment.assigned_at,
            "revoked_at": assignment.revoked_at,
        }
        with self._scope(write=True) as connection:
            connection.execute(
                pg_insert(self.acl)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=[self.acl.c.project_id, self.acl.c.principal_id, self.acl.c.role],
                    set_={
                        "granted_by": values["granted_by"],
                        "granted_at": values["granted_at"],
                        "revoked_at": values["revoked_at"],
                    },
                )
            )

    def revoke(self, project_id: ProjectId, principal_id: PrincipalId, role: ProjectRole) -> None:
        sa, _ = _sqlalchemy()
        with self._scope(write=True) as connection:
            connection.execute(
                sa.update(self.acl)
                .where(
                    self.acl.c.project_id == str(project_id),
                    self.acl.c.principal_id == str(principal_id),
                    self.acl.c.role == role.value,
                    self.acl.c.revoked_at.is_(None),
                )
                .values(revoked_at=datetime.now(UTC))
            )


__all__ = ["PostgresIdentityAclStore"]

