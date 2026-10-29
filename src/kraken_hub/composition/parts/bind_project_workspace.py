"""composition: bind project workspace."""
from __future__ import annotations
from pathlib import Path
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.workspace import (
    LayerFileBinding,
    LayerSourceScan,
    ProjectWorkspaceBinding,
    WorkspaceValidationError,
    project_workspace_to_dict,
)
from kraken_manager.application.ports import StorageProfile
from kraken_manager.domain.common import (
    LayerId,
    ProjectId,
)
from kraken_manager.domain.identity import (
    Permission,
    Principal,
)
from kraken_manager.domain.project import (
    Project,
)


class BindProjectWorkspaceMixin:
    def list_storage_profiles(self) -> tuple[StorageProfile, ...]:
        return (self.profile,)

    @property
    def supports_workspace_roots(self) -> bool:
        return True

    @property
    def supports_live_sync(self) -> bool:
        return False

    def project_storage_label(self, project_id: ProjectId | str) -> str:
        del project_id
        return "Локальный файл"

    def is_remote_project(self, project_id: ProjectId | str) -> bool:
        del project_id
        return False

    def project_workspace(self, project_id: ProjectId | str) -> ProjectWorkspaceBinding | None:
        return self.workspace_registry.get_project(str(project_id))

    def bind_project_workspace(
        self,
        *,
        principal: Principal,
        project: Project,
        source_root: Path | str,
        derived_root: Path | str,
        idempotency_key: str,
    ) -> ProjectWorkspaceBinding:
        """Give an existing project new workspace folders without moving its data."""
        current = self.get_project(project.id)
        if current is None:
            raise WorkspaceValidationError("project is no longer available")
        AuthorizationPolicy().require(
            principal=principal,
            storage=self.profile,
            permission=Permission.MANAGE_STRUCTURE,
            roles=self.project_roles(current.id, principal.id),
        )
        existing = self.project_workspace(current.id)
        if existing is not None:
            return existing
        binding = self.workspace_files.create_project(
            project_id=str(current.id),
            project_name=current.name,
            source_root=source_root,
            derived_root=derived_root,
        )
        try:
            self._record_workspace_event(
                principal=principal,
                project_id=current.id,
                stream_id=f"project-workspace:{current.id}",
                event_type="ProjectWorkspaceBoundV1",
                payload=project_workspace_to_dict(binding),
                idempotency_key=f"{idempotency_key}:workspace",
            )
        except Exception:
            self.workspace_registry.clear_project_binding(str(current.id))
            self.workspace_files.remove_project_layout(binding)
            raise
        return binding

    def layer_file_binding(
        self, project_id: ProjectId | str, layer_id: LayerId | str
    ) -> LayerFileBinding | None:
        return self.workspace_registry.get_layer(str(project_id), str(layer_id))

    def scan_layer_source(
        self, directory: Path | str, *, maximum_frames: int, project_id: ProjectId | str | None = None
    ) -> LayerSourceScan:
        del project_id
        return self.workspace_files.scan(directory, maximum_frames=maximum_frames)
