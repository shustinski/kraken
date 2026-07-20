"""Desktop composition root for authenticated local file projects."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kraken_manager.application.dto import (
    CommandContext,
    CreateLayerCommand,
    CreateProjectCommand,
)
from kraken_manager.application.ports import StorageProfile
from kraken_manager.application.use_cases import CreateLayerHandler, CreateProjectHandler
from kraken_manager.domain.common import LayerId, ProjectId
from kraken_manager.domain.identity import Performer, Principal
from kraken_manager.domain.project import GridOrientation, Layer, LayerType, Project, Representation
from kraken_manager.infrastructure.auth.identity_store import LocalIdentityAclStore
from kraken_manager.infrastructure.auth.local import LocalAccountStore, ScryptPasswordHasher
from kraken_manager.infrastructure.auth.performer_store import LocalSQLitePerformerStore
from kraken_manager.infrastructure.filesystem import (
    FileProjectLayout,
    FilesystemEventStore,
    LocalProjectUnitOfWorkFactory,
    SQLiteProjectionStore,
    filesystem_storage_profile,
)
from kraken_manager.infrastructure.projections import rebuild_filesystem_index


def default_data_dir() -> Path:
    override = os.environ.get("KRAKEN_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".kraken").resolve()


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class DesktopStorageProfiles:
    def __init__(self, *profiles: StorageProfile) -> None:
        self._profiles = {profile.id: profile for profile in profiles}

    def get(self, profile_id: str) -> StorageProfile | None:
        return self._profiles.get(profile_id)

    def list(self) -> tuple[StorageProfile, ...]:
        return tuple(self._profiles.values())


@dataclass(frozen=True, slots=True)
class DesktopSession:
    token: str
    principal: Principal
    expires_at: str


class EmbeddedProjectService:
    """Autonomous application service for machine-local file projects."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir or default_data_dir()).resolve()
        self.catalog_root = self.data_dir / "catalog"
        self.catalog_root.mkdir(parents=True, exist_ok=True)
        self.accounts = LocalAccountStore(self.data_dir / "accounts.sqlite3", ScryptPasswordHasher())
        self.identities = LocalIdentityAclStore(self.data_dir / "identity.sqlite3")
        self.performers = LocalSQLitePerformerStore(self.data_dir / "identity.sqlite3")
        self.profile = filesystem_storage_profile(str(self.catalog_root))
        self.profiles = DesktopStorageProfiles(self.profile)
        self.clock = SystemClock()

    @property
    def has_accounts(self) -> bool:
        return self.accounts.account_count() > 0

    def list_performers(self, *, include_archived: bool = False) -> tuple[Performer, ...]:
        return self.performers.list(include_archived=include_archived)

    def create_manual_performer(self, *, name: str, color: str) -> Performer:
        return self.performers.create(Performer.create(name=name, color=color))

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

    def create_project(
        self,
        *,
        principal: Principal,
        name: str,
        width: int,
        height: int,
        orientation: GridOrientation,
        idempotency_key: str,
        layer_template: bool = False,
    ) -> Project:
        command = CreateProjectCommand(
            context=CommandContext(actor=principal, idempotency_key=idempotency_key),
            name=name,
            width=width,
            height=height,
            orientation=orientation,
            storage_profile_id=self.profile.id,
        )
        assert command.project_id is not None
        project_id = str(command.project_id)
        project = CreateProjectHandler(self._uow(project_id), self.profiles, self.clock)(command)
        if layer_template:
            for order, layer_type in enumerate(LayerType, start=1):
                self.create_layer(
                    principal=principal,
                    project=project,
                    name=layer_type.value.capitalize(),
                    layer_type=layer_type,
                    order=order,
                    idempotency_key=f"{idempotency_key}:layer:{layer_type.value}",
                )
                project = self.get_project(project.id) or project
        return project

    def create_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        layer_type: LayerType,
        order: int,
        idempotency_key: str,
    ) -> Layer:
        command = CreateLayerCommand(
            context=CommandContext(actor=principal, idempotency_key=idempotency_key),
            project_id=project.id,
            name=name,
            type=layer_type,
            order=order,
            expected_project_revision=project.revision,
        )
        return CreateLayerHandler(self._uow(str(project.id)), self.profiles, self.clock)(command)

    def get_project(self, project_id: ProjectId | str, *, as_of: datetime | None = None) -> Project | None:
        projections = self._projection(project_id)
        return projections.get_project(ProjectId(str(project_id)), as_of=as_of)

    def _projection(self, project_id: ProjectId | str) -> SQLiteProjectionStore:
        identifier = str(project_id)
        events = FilesystemEventStore(self.catalog_root, identifier)
        with events.lock.hold(events.lock_timeout):
            projections = SQLiteProjectionStore(self.catalog_root, identifier)
            if projections.checkpoint != events.last_global_position():
                rebuild_filesystem_index(events, projections, acl=self.identities)
                projections = SQLiteProjectionStore(self.catalog_root, identifier)
        return projections

    def list_projects(self, *, include_archived: bool = False) -> tuple[Project, ...]:
        projects_root = self.catalog_root / "projects"
        if not projects_root.is_dir():
            return ()
        result: list[Project] = []
        for directory in sorted(path for path in projects_root.iterdir() if path.is_dir()):
            descriptor = directory / "project.json"
            try:
                project_id = ProjectId(
                    str(json.loads(descriptor.read_text(encoding="utf-8"))["project_id"])
                    if descriptor.is_file()
                    else directory.name
                )
                project = self._projection(project_id).get_project(project_id)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if project is not None and not descriptor.is_file():
                FileProjectLayout(self.catalog_root, str(project.id)).initialize(
                    {
                        "project_id": str(project.id),
                        "name": project.name,
                        "width": project.width,
                        "height": project.height,
                        "orientation": project.orientation.value,
                        "storage_profile": project.storage_profile,
                    }
                )
            if project is not None and (include_archived or project.state.value != "archived"):
                result.append(project)
        return tuple(sorted(result, key=lambda project: project.name.casefold()))

    def list_layers(
        self, project_id: ProjectId | str, *, as_of: datetime | None = None
    ) -> tuple[Layer, ...]:
        return self._projection(project_id).list_layers(ProjectId(str(project_id)), as_of=as_of)

    def list_representations(
        self, project_id: ProjectId | str, layer_id: LayerId | str, *, as_of: datetime | None = None
    ) -> tuple[Representation, ...]:
        return self._projection(project_id).list_representations(LayerId(str(layer_id)), as_of=as_of)

    def history(self, project_id: ProjectId | str, *, as_of: datetime | None = None) -> tuple[object, ...]:
        store = FilesystemEventStore(self.catalog_root, str(project_id))
        events = []
        for stored in store.iter_project():
            event = store._decode(stored)
            if as_of is None or getattr(event, "recorded_at", self.clock.now()) <= as_of:
                events.append(event)
        return tuple(events)

    def attach_project(self, project_directory: Path | str) -> Project:
        source = Path(project_directory).resolve(strict=True)
        descriptor = json.loads((source / "project.json").read_text(encoding="utf-8"))
        project_id = str(descriptor["project_id"])
        expected = FileProjectLayout(self.catalog_root, project_id).project_dir.resolve()
        if source != expected:
            raise ValueError("v1 attaches only projects already located in the configured local catalog")
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project cannot be reconstructed from its authoritative event log")
        return project


__all__ = ["DesktopSession", "DesktopStorageProfiles", "EmbeddedProjectService", "SystemClock", "default_data_dir"]
