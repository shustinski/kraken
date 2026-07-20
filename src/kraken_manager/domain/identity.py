"""Identity, performers, and Kraken-owned role definitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from .common import (
    DomainValidationError,
    PerformerId,
    PrincipalId,
    ProjectId,
    as_utc,
    new_uuid,
    require_non_empty,
    utc_now,
    validate_uuid,
)


class PrincipalProvider(StrEnum):
    LOCAL = "local"
    GITLAB = "gitlab"


class SystemRole(StrEnum):
    SERVER_ADMIN = "server_admin"


class ProjectRole(StrEnum):
    OWNER = "owner"
    MANAGER = "manager"
    CONTRIBUTOR = "contributor"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


class Permission(StrEnum):
    VIEW_PROJECT = "view_project"
    VIEW_HISTORY = "view_history"
    EXPORT_STATISTICS = "export_statistics"
    RENAME_PROJECT = "rename_project"
    ARCHIVE_PROJECT = "archive_project"
    MIGRATE_PROJECT = "migrate_project"
    MANAGE_ACL = "manage_acl"
    MANAGE_STRUCTURE = "manage_structure"
    ASSIGN_WORK = "assign_work"
    MANAGE_REVIEW = "manage_review"
    ACCEPT_REVIEW = "accept_review"
    IMPORT_ARTIFACT = "import_artifact"
    RUN_PLUGIN = "run_plugin"
    ADD_NOTE = "add_note"
    RETURN_REVIEW = "return_review"


_VIEW: Final = frozenset(
    {Permission.VIEW_PROJECT, Permission.VIEW_HISTORY, Permission.EXPORT_STATISTICS}
)
ROLE_PERMISSIONS: Final = MappingProxyType(
    {
        ProjectRole.VIEWER: _VIEW,
        ProjectRole.REVIEWER: _VIEW | {Permission.ADD_NOTE, Permission.RETURN_REVIEW},
        ProjectRole.CONTRIBUTOR: _VIEW
        | {Permission.IMPORT_ARTIFACT, Permission.RUN_PLUGIN, Permission.ADD_NOTE, Permission.RETURN_REVIEW},
        ProjectRole.MANAGER: _VIEW
        | {
            Permission.MANAGE_STRUCTURE,
            Permission.ASSIGN_WORK,
            Permission.MANAGE_REVIEW,
            Permission.ACCEPT_REVIEW,
            Permission.IMPORT_ARTIFACT,
            Permission.RUN_PLUGIN,
            Permission.ADD_NOTE,
            Permission.RETURN_REVIEW,
            Permission.RENAME_PROJECT,
        },
        ProjectRole.OWNER: frozenset(Permission),
    }
)


@dataclass(frozen=True, slots=True)
class Principal:
    """Authenticated identity; GitLab is an identity provider, not an ACL."""

    id: PrincipalId
    provider: PrincipalProvider
    subject: str
    display_name: str
    issuer: str | None = None
    email: str | None = None
    active: bool = True
    system_roles: frozenset[SystemRole] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", PrincipalId(validate_uuid(str(self.id), field="principal.id")))
        if not isinstance(self.provider, PrincipalProvider):
            object.__setattr__(self, "provider", PrincipalProvider(self.provider))
        object.__setattr__(self, "subject", require_non_empty(self.subject, field="principal.subject", maximum=512))
        object.__setattr__(
            self, "display_name", require_non_empty(self.display_name, field="principal.display_name", maximum=255)
        )
        if self.provider is PrincipalProvider.GITLAB:
            if self.issuer is None:
                raise DomainValidationError("GitLab principal requires an OIDC issuer")
            object.__setattr__(self, "issuer", require_non_empty(self.issuer, field="principal.issuer", maximum=2048))
        elif self.issuer is not None:
            raise DomainValidationError("local principal must not have an OIDC issuer")
        if self.email is not None:
            email = require_non_empty(self.email, field="principal.email", maximum=320)
            if "@" not in email:
                raise DomainValidationError("principal.email must look like an email address")
            object.__setattr__(self, "email", email)
        roles = frozenset(SystemRole(role) for role in self.system_roles)
        object.__setattr__(self, "system_roles", roles)

    @classmethod
    def local(
        cls,
        *,
        subject: str,
        display_name: str,
        principal_id: PrincipalId | str | None = None,
        email: str | None = None,
    ) -> Principal:
        return cls(
            id=PrincipalId(str(principal_id) if principal_id is not None else new_uuid()),
            provider=PrincipalProvider.LOCAL,
            subject=subject,
            display_name=display_name,
            email=email,
        )

    @classmethod
    def gitlab(
        cls,
        *,
        issuer: str,
        subject: str,
        display_name: str,
        principal_id: PrincipalId | str | None = None,
        email: str | None = None,
    ) -> Principal:
        return cls(
            id=PrincipalId(str(principal_id) if principal_id is not None else new_uuid()),
            provider=PrincipalProvider.GITLAB,
            issuer=issuer,
            subject=subject,
            display_name=display_name,
            email=email,
        )

    @property
    def external_key(self) -> str:
        if self.provider is PrincipalProvider.GITLAB:
            return f"gitlab:{self.issuer}:{self.subject}"
        return f"local:{self.subject}"


@dataclass(frozen=True, slots=True)
class Performer:
    """Attribution target, optionally linked to an authenticating principal."""

    id: PerformerId
    name: str
    color: str
    principal_id: PrincipalId | None = None
    active: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", PerformerId(validate_uuid(str(self.id), field="performer.id")))
        object.__setattr__(self, "name", require_non_empty(self.name, field="performer.name", maximum=255))
        color = self.color.strip().upper()
        if len(color) != 7 or not color.startswith("#") or any(character not in "0123456789ABCDEF" for character in color[1:]):
            raise DomainValidationError("performer.color must be in #RRGGBB format")
        object.__setattr__(self, "color", color)
        if self.principal_id is not None:
            object.__setattr__(
                self,
                "principal_id",
                PrincipalId(validate_uuid(str(self.principal_id), field="performer.principal_id")),
            )

    @classmethod
    def create(
        cls,
        *,
        name: str,
        color: str,
        principal_id: PrincipalId | None = None,
        performer_id: PerformerId | str | None = None,
    ) -> Performer:
        return cls(
            id=PerformerId(str(performer_id) if performer_id is not None else new_uuid()),
            name=name,
            color=color,
            principal_id=principal_id,
        )

    def archive(self) -> Performer:
        return self if not self.active else replace(self, active=False)


@dataclass(frozen=True, slots=True)
class ProjectRoleAssignment:
    project_id: ProjectId
    principal_id: PrincipalId
    role: ProjectRole
    assigned_by: PrincipalId
    assigned_at: datetime
    revoked_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "project_id", ProjectId(validate_uuid(str(self.project_id), field="role_assignment.project_id"))
        )
        object.__setattr__(
            self,
            "principal_id",
            PrincipalId(validate_uuid(str(self.principal_id), field="role_assignment.principal_id")),
        )
        if not isinstance(self.role, ProjectRole):
            object.__setattr__(self, "role", ProjectRole(self.role))
        object.__setattr__(
            self,
            "assigned_by",
            PrincipalId(validate_uuid(str(self.assigned_by), field="role_assignment.assigned_by")),
        )
        object.__setattr__(
            self, "assigned_at", as_utc(self.assigned_at, field="role_assignment.assigned_at")
        )
        if self.revoked_at is not None:
            revoked_at = as_utc(self.revoked_at, field="role_assignment.revoked_at")
            if revoked_at < self.assigned_at:
                raise DomainValidationError("role revocation cannot precede assignment")
            object.__setattr__(self, "revoked_at", revoked_at)

    @classmethod
    def create(
        cls,
        *,
        project_id: ProjectId,
        principal_id: PrincipalId,
        role: ProjectRole,
        assigned_by: PrincipalId,
        assigned_at: datetime | None = None,
    ) -> ProjectRoleAssignment:
        return cls(
            project_id=project_id,
            principal_id=principal_id,
            role=role,
            assigned_by=assigned_by,
            assigned_at=assigned_at or utc_now(),
        )

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    def revoke(self, at: datetime | None = None) -> ProjectRoleAssignment:
        if not self.active:
            return self
        return replace(self, revoked_at=at or utc_now())


def permissions_for_roles(roles: frozenset[ProjectRole] | set[ProjectRole]) -> frozenset[Permission]:
    permissions: set[Permission] = set()
    for role in roles:
        permissions.update(ROLE_PERMISSIONS[ProjectRole(role)])
    return frozenset(permissions)


__all__ = [
    "Permission",
    "Performer",
    "Principal",
    "PrincipalProvider",
    "ProjectRole",
    "ProjectRoleAssignment",
    "ROLE_PERMISSIONS",
    "SystemRole",
    "permissions_for_roles",
]
