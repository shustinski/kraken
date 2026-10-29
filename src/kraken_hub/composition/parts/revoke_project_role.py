"""composition: revoke project role."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from kraken_manager.application.dto import (
    AssignProjectRoleCommand,
    CommandContext,
    RevokeProjectRoleCommand,
)
from kraken_manager.application.acl import AssignProjectRoleHandler, RevokeProjectRoleHandler
from kraken_manager.domain.common import (
    PerformerId,
    PrincipalId,
    ProjectId,
)
from kraken_manager.domain.identity import (
    Permission,
    Performer,
    Principal,
    ProjectRole,
    ROLE_PERMISSIONS,
)
from kraken_manager.domain.project import (
    Project,
)
from kraken_manager.infrastructure.auth.identity_store import LocalIdentityAclStore
from kraken_manager.infrastructure.auth.local import LocalAccountStore, ScryptPasswordHasher
from kraken_manager.infrastructure.auth.performer_store import LocalSQLitePerformerStore
from kraken_manager.infrastructure.filesystem import (
    FilesystemEventStore,
    LocalProjectUnitOfWorkFactory,
    filesystem_storage_profile,
)
from kraken_manager.infrastructure.workspace_files import WorkspaceFileService, WorkspaceRegistry
from kraken_hub.infrastructure.credentials.secret_store import KeyringSecretStore

from kraken_hub.composition.parts.frame_cell_snapshot import (
    DesktopSession,
    DesktopStorageProfiles,
    SystemClock,
    default_data_dir,
)


class RevokeProjectRoleMixin:
    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        secret_store: object | None = None,
    ) -> None:
        self.data_dir = Path(data_dir or default_data_dir()).resolve()
        self.catalog_root = self.data_dir / "catalog"
        self.catalog_root.mkdir(parents=True, exist_ok=True)
        self.accounts = LocalAccountStore(self.data_dir / "accounts.sqlite3", ScryptPasswordHasher())
        self.identities = LocalIdentityAclStore(self.data_dir / "identity.sqlite3")
        self.performers = LocalSQLitePerformerStore(self.data_dir / "identity.sqlite3")
        self.profile = filesystem_storage_profile(str(self.catalog_root))
        self.workspace_registry = WorkspaceRegistry(self.catalog_root)
        self.workspace_files = WorkspaceFileService(self.workspace_registry)
        self.default_source_root = self.data_dir / "workspace-source"
        self.default_derived_root = self.data_dir / "workspace-derived"
        self.default_source_root.mkdir(parents=True, exist_ok=True)
        self.default_derived_root.mkdir(parents=True, exist_ok=True)
        self._external_source_indexes: dict[
            tuple[str, str, int], tuple[Path | None, ...]
        ] = {}
        self.profiles = DesktopStorageProfiles(self.profile)
        self.clock = SystemClock()
        self.secrets = secret_store or KeyringSecretStore()

    @property
    def has_accounts(self) -> bool:
        return self.accounts.account_count() > 0

    def create_initial_account(self, username: str, display_name: str, password: str) -> DesktopSession:
        """Create and sign in the first workstation-local account."""
        if self.has_accounts:
            raise ValueError("A local account already exists")
        account = self.accounts.create_account(username, display_name, password)
        principal = Principal.local(
            subject=account.username,
            display_name=account.display_name,
            principal_id=account.account_id,
        )
        self.identities.save(principal)
        session = self.accounts.authenticate(account.username, password)
        if session is None:  # pragma: no cover - the freshly created account is enabled
            raise RuntimeError("The new local account could not be signed in")
        return DesktopSession(session.token, principal, session.expires_at)

    def list_performers(self, *, include_archived: bool = False) -> tuple[Performer, ...]:
        return self.performers.list(include_archived=include_archived)

    def create_manual_performer(self, *, name: str, color: str) -> Performer:
        return self.performers.create(Performer.create(name=name, color=color))

    def update_performer(
        self,
        *,
        performer_id: PerformerId | str,
        name: str,
        color: str,
    ) -> Performer:
        performer = self.performers.get(PerformerId(str(performer_id)))
        if performer is None:
            raise ValueError(f"Performer {performer_id} was not found")
        return self.performers.update(replace(performer, name=name, color=color))

    def archive_performer(self, performer_id: PerformerId | str) -> Performer:
        return self.performers.archive(PerformerId(str(performer_id)))

    def project_roles(
        self, project_id: ProjectId | str, principal_id: PrincipalId | str
    ) -> frozenset[ProjectRole]:
        return self.identities.roles_for(ProjectId(str(project_id)), PrincipalId(str(principal_id)))

    def list_principals(self, *, include_inactive: bool = False) -> tuple[Principal, ...]:
        return self.identities.list(include_inactive=include_inactive)

    def list_project_principals(
        self,
        project_id: ProjectId | str,
        *,
        include_inactive: bool = False,
    ) -> tuple[Principal, ...]:
        project = ProjectId(str(project_id))
        return tuple(
            principal
            for principal in self.list_principals(include_inactive=include_inactive)
            if self.identities.roles_for(project, principal.id)
        )

    def project_role_revision(
        self,
        project_id: ProjectId | str,
        principal_id: PrincipalId | str,
    ) -> int:
        return FilesystemEventStore(
            self.catalog_root,
            str(project_id),
        ).current_revision(f"acl:{project_id}:{principal_id}")

    def project_permissions(
        self,
        project_id: ProjectId | str,
        principal: Principal,
    ) -> frozenset[Permission]:
        permissions: set[Permission] = set()
        for role in self.project_roles(project_id, principal.id):
            permissions.update(ROLE_PERMISSIONS[role])
        return frozenset(permissions)

    def assign_project_role(
        self,
        *,
        principal: Principal,
        project: Project,
        target_principal_id: PrincipalId,
        role: ProjectRole,
        expected_revision: int,
        idempotency_key: str,
    ) -> frozenset[ProjectRole]:
        return AssignProjectRoleHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            AssignProjectRoleCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id,
                principal_id=target_principal_id,
                role=role,
                expected_revision=expected_revision,
            )
        )

    def revoke_project_role(
        self,
        *,
        principal: Principal,
        project: Project,
        target_principal_id: PrincipalId,
        role: ProjectRole,
        expected_revision: int,
        idempotency_key: str,
    ) -> frozenset[ProjectRole]:
        if role is ProjectRole.MAINTAINER:
            owners = tuple(
                candidate
                for candidate in self.identities.list()
                if ProjectRole.MAINTAINER
                in self.identities.roles_for(project.id, candidate.id)
            )
            if len(owners) <= 1 and any(
                candidate.id == target_principal_id for candidate in owners
            ):
                raise ValueError("Нельзя отозвать роль последнего сопровождающего проекта.")
        return RevokeProjectRoleHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            RevokeProjectRoleCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id,
                principal_id=target_principal_id,
                role=role,
                expected_revision=expected_revision,
            )
        )

    def login(self, username: str, password: str) -> DesktopSession | None:
        session = self.accounts.authenticate(username, password)
        if session is None:
            return None
        principal = self.identities.get_by_external_key(f"local:{session.account.username}")
        if principal is None:
            principal = Principal.local(
                subject=session.account.username,
                display_name=session.account.display_name,
                principal_id=session.account.account_id,
            )
            self.identities.save(principal)
        return DesktopSession(session.token, principal, session.expires_at)

    def resolve_session(self, token: str) -> Principal | None:
        account = self.accounts.resolve_session(token)
        if account is None:
            return None
        return self.identities.get(PrincipalId(account.account_id))

    def _uow(self, project_id: str) -> LocalProjectUnitOfWorkFactory:
        return LocalProjectUnitOfWorkFactory(
            self.catalog_root,
            project_id,
            identities=self.identities,
            acl=self.identities,
        )
