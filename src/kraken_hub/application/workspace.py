"""Storage-agnostic workspace façade used by the desktop UI."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from kraken_hub.application.desktop_operations import DesktopOperations
from kraken_hub.application.notifications import ProjectEventWake, ServerConnectionSummary
from kraken_manager.application.dto import StorageBackendKind, StorageScope
from kraken_manager.application.ports import StorageCapabilities, StorageProfile
from kraken_manager.application.project_operations import ProjectOperations

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
        max_frames=None,
    ),
)


@runtime_checkable
class ProjectWorkspaceService(ProjectOperations, DesktopOperations, Protocol):
    """Desktop configuration plus the complete application operation contract."""

    profile: StorageProfile
    data_dir: Path

    @property
    def local_administration(self) -> ProjectWorkspaceService: ...

    @property
    def supports_server_connection(self) -> bool: ...

    @property
    def server_connection(self) -> ServerConnectionSummary | None: ...

    @property
    def supports_workspace_roots(self) -> bool: ...

    @property
    def supports_live_sync(self) -> bool: ...


__all__ = [
    "ProjectEventWake",
    "ProjectWorkspaceService",
    "REMOTE_STORAGE_PROFILE",
]
