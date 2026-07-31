"""Storage-agnostic workspace façade used by the desktop UI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from kraken_manager.application.ports import StorageCapabilities, StorageProfile
from kraken_manager.application.dto import StorageBackendKind, StorageScope
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.project import GridOrientation, Project, ProjectState


REMOTE_STORAGE_PROFILE = StorageProfile(
    id="server-postgres",
    name="Kraken Server PostgreSQL",
    metadata_backend=StorageBackendKind.POSTGRESQL,
    blob_backend="filesystem",
    scope=StorageScope.SHARED,
    capabilities=StorageCapabilities(
        multi_writer=True,
        transactions=True,
        snapshots=True,
        streaming=True,
        external_references=True,
        max_frames=1_000_000,
    ),
)


@dataclass(frozen=True, slots=True)
class ProjectEventWake:
    """Compact wake signal from shared storage (WebSocket / outbox)."""

    project_id: str
    event_type: str
    event_id: str
    position: int | None = None
    revision: int | None = None


@runtime_checkable
class ProjectWorkspaceService(Protocol):
    """Minimal contract the desktop UI relies on for catalog + project ops."""

    profile: StorageProfile
    data_dir: Path

    @property
    def supports_workspace_roots(self) -> bool: ...

    @property
    def supports_live_sync(self) -> bool: ...

    def list_storage_profiles(self) -> tuple[StorageProfile, ...]: ...

    def list_projects(self, *, include_archived: bool = False) -> tuple[Project, ...]: ...

    def get_project(
        self,
        project_id: object,
        *,
        as_of: datetime | None = None,
    ) -> Project | None: ...

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
    ) -> Project: ...

    def project_storage_label(self, project_id: object) -> str: ...

    def is_remote_project(self, project_id: object) -> bool: ...

    def project_workspace(self, project_id: object) -> object | None: ...


def project_from_server_dict(payload: Mapping[str, object]) -> Project:
    created = payload.get("created_at")
    if isinstance(created, str):
        created_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
    elif isinstance(created, datetime):
        created_at = created
    else:
        created_at = datetime.now().astimezone()
    return Project(
        id=str(payload.get("project_id") or payload.get("id")),
        name=str(payload["name"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        orientation=GridOrientation(str(payload.get("orientation", GridOrientation.Y_DOWN.value))),
        storage_profile=str(payload.get("storage_profile") or REMOTE_STORAGE_PROFILE.id),
        state=ProjectState(str(payload.get("state", ProjectState.ACTIVE.value))),
        revision=int(payload.get("revision", 0)),
        created_at=created_at,
    )


__all__ = [
    "ProjectEventWake",
    "ProjectWorkspaceService",
    "REMOTE_STORAGE_PROFILE",
    "project_from_server_dict",
]
