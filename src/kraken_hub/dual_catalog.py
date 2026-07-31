"""Compose local filesystem and remote server catalogs behind one façade."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from kraken_manager.application.ports import StorageProfile
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.project import GridOrientation, Project

from .composition import EmbeddedProjectService
from .remote_client import RemoteServerProjectService
from .workspace_service import REMOTE_STORAGE_PROFILE, ProjectEventWake


class DualCatalogService:
    """Route catalog/project operations to local or remote backends by ownership."""

    def __init__(
        self,
        local: EmbeddedProjectService,
        remote: RemoteServerProjectService | None = None,
    ) -> None:
        self.local = local
        self.remote = remote
        self.data_dir = local.data_dir
        self.profile = local.profile
        self._remote_ids: set[str] = set()
        if remote is not None:
            try:
                for project in remote.list_projects(include_archived=True):
                    self._remote_ids.add(str(project.id))
            except Exception:
                # Server may be temporarily unreachable; catalog refresh will retry.
                self._remote_ids = set()

    @property
    def supports_workspace_roots(self) -> bool:
        return True

    @property
    def supports_live_sync(self) -> bool:
        return self.remote is not None and self.remote.supports_live_sync

    @property
    def has_accounts(self) -> bool:
        return self.local.has_accounts

    @property
    def accounts(self) -> Any:
        return self.local.accounts

    @property
    def identities(self) -> Any:
        return self.local.identities

    @property
    def performers(self) -> Any:
        return self.local.performers

    @property
    def default_source_root(self) -> Path:
        return self.local.default_source_root

    @property
    def default_derived_root(self) -> Path:
        return self.local.default_derived_root

    @property
    def workspace_files(self) -> Any:
        return self.local.workspace_files

    @property
    def workspace_registry(self) -> Any:
        return self.local.workspace_registry

    def list_storage_profiles(self) -> tuple[StorageProfile, ...]:
        profiles = [self.local.profile]
        if self.remote is not None:
            profiles.append(REMOTE_STORAGE_PROFILE)
        return tuple(profiles)

    def is_remote_project(self, project_id: object) -> bool:
        return str(project_id) in self._remote_ids

    def project_storage_label(self, project_id: object) -> str:
        if self.is_remote_project(project_id):
            return "Сервер PostgreSQL"
        return "Локальный файл"

    def _backend(self, project_id: object) -> Any:
        if self.is_remote_project(project_id) and self.remote is not None:
            return self.remote
        return self.local

    def list_projects(self, *, include_archived: bool = False) -> tuple[Project, ...]:
        local_projects = list(self.local.list_projects(include_archived=include_archived))
        remote_projects: list[Project] = []
        if self.remote is not None:
            try:
                remote_projects = list(self.remote.list_projects(include_archived=include_archived))
                self._remote_ids = {
                    str(project.id)
                    for project in self.remote.list_projects(include_archived=True)
                }
            except Exception:
                remote_projects = []
        merged = {str(project.id): project for project in local_projects}
        for project in remote_projects:
            merged[str(project.id)] = project
        return tuple(sorted(merged.values(), key=lambda item: item.name.casefold()))

    def get_project(
        self,
        project_id: object,
        *,
        as_of: datetime | None = None,
    ) -> Project | None:
        if self.is_remote_project(project_id) and self.remote is not None:
            return self.remote.get_project(project_id, as_of=as_of)
        project = self.local.get_project(project_id, as_of=as_of)
        if project is not None:
            return project
        if self.remote is not None:
            remote = self.remote.get_project(project_id, as_of=as_of)
            if remote is not None:
                self._remote_ids.add(str(remote.id))
            return remote
        return None

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
        source_root: Path | str | None = None,
        derived_root: Path | str | None = None,
        storage_profile_id: str | None = None,
    ) -> Project:
        profile_id = storage_profile_id or self.local.profile.id
        if profile_id == REMOTE_STORAGE_PROFILE.id:
            if self.remote is None:
                raise RuntimeError(
                    "Shared PostgreSQL catalog is not configured. "
                    "Set KRAKEN_SERVER_URL and KRAKEN_GITLAB_TOKEN."
                )
            actor = self.remote.auth.principal
            project = self.remote.create_project(
                principal=actor,
                name=name,
                width=width,
                height=height,
                orientation=orientation,
                idempotency_key=idempotency_key,
                storage_profile_id=profile_id,
            )
            self._remote_ids.add(str(project.id))
            return project
        return self.local.create_project(
            principal=principal,
            name=name,
            width=width,
            height=height,
            orientation=orientation,
            idempotency_key=idempotency_key,
            layer_template=layer_template,
            source_root=source_root,
            derived_root=derived_root,
        )

    def project_workspace(self, project_id: object) -> object | None:
        if self.is_remote_project(project_id):
            return None
        return self.local.project_workspace(project_id)

    def start_sync(self) -> None:
        if self.remote is not None:
            self.remote.start_sync()

    def stop_sync(self) -> None:
        if self.remote is not None:
            self.remote.stop_sync()

    def subscribe_project(self, project_id: object) -> None:
        if self.is_remote_project(project_id) and self.remote is not None:
            self.remote.subscribe_project(project_id)

    def unsubscribe_project(self, project_id: object) -> None:
        if self.remote is not None:
            self.remote.unsubscribe_project(project_id)

    def add_wake_handler(self, handler: Callable[[ProjectEventWake], None]) -> None:
        if self.remote is not None:
            self.remote.add_wake_handler(handler)

    def add_catalog_handler(self, handler: Callable[[], None]) -> None:
        if self.remote is not None:
            self.remote.add_catalog_handler(handler)

    def rename_project(self, **kwargs):
        project = kwargs.get("project")
        if project is not None and self.is_remote_project(project.id) and self.remote is not None:
            return self.remote.rename_project(**kwargs)
        return self.local.rename_project(**kwargs)

    def archive_project(self, **kwargs):
        project = kwargs.get("project")
        if project is not None and self.is_remote_project(project.id) and self.remote is not None:
            return self.remote.archive_project(**kwargs)
        return self.local.archive_project(**kwargs)

    def restore_project(self, **kwargs):
        project = kwargs.get("project")
        if project is not None and self.is_remote_project(project.id) and self.remote is not None:
            return self.remote.restore_project(**kwargs)
        return self.local.restore_project(**kwargs)

    def list_layers(self, project_id, *, include_archived: bool = False):
        return self._backend(project_id).list_layers(project_id, include_archived=include_archived)

    def project_permissions(self, project_id, principal):
        backend = self._backend(project_id)
        if backend is self.remote and self.remote is not None:
            return self.remote.project_permissions(project_id, self.remote.auth.principal)
        return self.local.project_permissions(project_id, principal)

    def history(self, project_id, *, as_of=None):
        return self._backend(project_id).history(project_id, as_of=as_of)

    def matrix_viewport(self, project_id, **kwargs):
        backend = self._backend(project_id)
        if backend is self.remote and self.remote is not None:
            return self.remote.matrix_viewport(
                project_id,
                layer_id=kwargs["layer_id"],
                x1=kwargs["x1"],
                y1=kwargs["y1"],
                x2=kwargs["x2"],
                y2=kwargs["y2"],
                lod=kwargs.get("lod", 0),
                representation_ids=kwargs.get("representation_ids", ()),
            )
        return self.local.matrix_viewport(project_id, **kwargs)

    def create_initial_account(self, username: str, display_name: str, password: str):
        return self.local.create_initial_account(username, display_name, password)

    def login(self, username: str, password: str):
        return self.local.login(username, password)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.local, name)


def build_workspace_service(
    *,
    data_dir: Path | str | None = None,
) -> DualCatalogService | EmbeddedProjectService:
    local = EmbeddedProjectService(data_dir=data_dir)
    from .remote_client import RemoteServerError, build_remote_service

    try:
        remote = build_remote_service(data_dir=local.data_dir)
    except Exception:
        remote = None
    if remote is None:
        return local
    try:
        return DualCatalogService(local, remote)
    except RemoteServerError:
        return local


__all__ = ["DualCatalogService", "build_workspace_service"]
