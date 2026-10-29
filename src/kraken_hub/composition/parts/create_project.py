"""composition: create project."""
from __future__ import annotations
from pathlib import Path
from kraken_manager.application.dto import (
    CommandContext,
    CreateProjectCommand,
)
from kraken_manager.workspace import (
    project_workspace_to_dict,
    validate_workspace_name,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    GridOrientation,
    LayerType,
    Project,
)


class CreateProjectMixin:
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
        if storage_profile_id not in {None, self.profile.id}:
            raise ValueError(
                f"Local catalog only supports storage profile {self.profile.id!r}"
            )
        safe_name = validate_workspace_name(name, field_name="Название проекта")
        command = CreateProjectCommand(
            context=CommandContext(actor=principal, idempotency_key=idempotency_key),
            name=safe_name,
            width=width,
            height=height,
            orientation=orientation,
            storage_profile_id=self.profile.id,
        )
        assert command.project_id is not None
        project_id = str(command.project_id)
        binding = self.workspace_files.create_project(
            project_id=project_id,
            project_name=safe_name,
            source_root=source_root or self.default_source_root,
            derived_root=derived_root or self.default_derived_root,
        )

        try:
            project = self._commands(project_id).execute(command)
        except Exception:
            self.workspace_files.remove_project_layout(binding)
            self.workspace_registry.remove_project(project_id)
            raise
        self._record_workspace_event(
            principal=principal,
            project_id=project.id,
            stream_id=f"project-workspace:{project.id}",
            event_type="ProjectWorkspaceBoundV1",
            payload=project_workspace_to_dict(binding),
            idempotency_key=f"{idempotency_key}:workspace",
        )
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
