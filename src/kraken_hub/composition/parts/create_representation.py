"""composition: create representation."""
from __future__ import annotations
from pathlib import Path
from kraken_core.safe_files import ensure_regular_directory
from kraken_manager.application.dto import (
    ActivateRepresentationCommand,
    ArchiveRepresentationCommand,
    CommandContext,
    CreateLayerCommand,
    CreateRepresentationCommand,
    DeactivateRepresentationCommand,
    RenameRepresentationCommand,
    UpdateRepresentationNoteCommand,
)
from kraken_manager.application.imports import ImportMappingMode, ImportPlan, ImportPlanner, ImportSource
from kraken_manager.workspace import (
    DerivedRun,
)
from kraken_manager.domain.common import (
    LayerId,
    ProjectId,
    new_uuid,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    Layer,
    LayerType,
    Project,
    Representation,
    RepresentationKind,
    RepresentationPurpose,
)


class CreateRepresentationMixin:
    def fail_derived_run(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        run_id: str,
        error: str,
    ) -> DerivedRun:
        failed = self.workspace_files.fail_run(
            project_id=str(project_id),
            run_id=run_id,
            error=error,
        )
        self._record_workspace_event(
            principal=principal,
            project_id=project_id,
            stream_id=f"derived-run:{failed.run_id}",
            event_type="DerivedRunFailedV1",
            payload={
                "run_id": failed.run_id,
                "layer_id": failed.layer_id,
                "kind": failed.kind.value,
                "state": failed.state.value,
                "plugin_id": failed.plugin_id,
                "operation": failed.operation,
                "error": str(error)[:10_000],
            },
            idempotency_key=f"derived-run:{failed.run_id}:failed",
        )
        return failed

    def create_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        layer_type: LayerType,
        order: int,
        idempotency_key: str,
        layer_id: LayerId | str | None = None,
    ) -> Layer:
        command = CreateLayerCommand(
            context=CommandContext(actor=principal, idempotency_key=idempotency_key),
            project_id=project.id,
            name=name,
            type=layer_type,
            order=order,
            expected_project_revision=project.revision,
            layer_id=LayerId(str(layer_id)) if layer_id is not None else LayerId(new_uuid()),
        )
        return self._commands(str(project.id)).execute(command)

    def create_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        name: str,
        kind: RepresentationKind,
        idempotency_key: str,
        note: str = "",
        source: str | None = None,
        source_image_representation_id=None,
        active: bool = False,
        purpose: RepresentationPurpose = RepresentationPurpose.SOURCE,
    ) -> Representation:
        current_layer = next((item for item in self.list_layers(project.id) if item.id == layer.id), None)
        if current_layer is None:
            raise ValueError("Layer is no longer available")
        command = CreateRepresentationCommand(
            context=CommandContext(actor=principal, idempotency_key=idempotency_key),
            project_id=project.id,
            layer_id=current_layer.id,
            name=name,
            kind=kind,
            expected_layer_revision=current_layer.revision,
            note=note,
            source=source,
            source_image_representation_id=source_image_representation_id,
            active=active,
            purpose=purpose,
        )
        return self._commands(str(project.id)).execute(command)

    def rename_representation(
        self, *, principal: Principal, project: Project, layer: Layer,
        representation: Representation, name: str, idempotency_key: str
    ) -> Representation:
        return self._commands(str(project.id)).execute(
            RenameRepresentationCommand(
                CommandContext(actor=principal, idempotency_key=idempotency_key),
                project.id, layer.id, representation.id, name, layer.revision, representation.revision,
            )
        )

    def update_representation_note(
        self, *, principal: Principal, project: Project, layer: Layer,
        representation: Representation, note: str, idempotency_key: str
    ) -> Representation:
        return self._commands(str(project.id)).execute(
            UpdateRepresentationNoteCommand(
                CommandContext(actor=principal, idempotency_key=idempotency_key),
                project.id, layer.id, representation.id, note, layer.revision, representation.revision,
            )
        )

    def activate_representation(
        self, *, principal: Principal, project: Project, layer: Layer,
        representation: Representation, idempotency_key: str
    ) -> Representation:
        return self._commands(str(project.id)).execute(
            ActivateRepresentationCommand(
                CommandContext(actor=principal, idempotency_key=idempotency_key),
                project.id, layer.id, representation.id, layer.revision, representation.revision,
            )
        )

    def deactivate_representation(
        self, *, principal: Principal, project: Project, layer: Layer,
        representation: Representation, idempotency_key: str
    ) -> Representation:
        return self._commands(str(project.id)).execute(
            DeactivateRepresentationCommand(
                CommandContext(actor=principal, idempotency_key=idempotency_key),
                project.id,
                layer.id,
                representation.id,
                layer.revision,
                representation.revision,
            )
        )

    def archive_representation(
        self, *, principal: Principal, project: Project, layer: Layer,
        representation: Representation, idempotency_key: str
    ) -> Representation:
        return self._commands(str(project.id)).execute(
            ArchiveRepresentationCommand(
                CommandContext(actor=principal, idempotency_key=idempotency_key),
                project.id, layer.id, representation.id, layer.revision, representation.revision,
            )
        )

    def plan_import_directory(
        self,
        *,
        project: Project,
        directory: Path | str,
        mode: ImportMappingMode = ImportMappingMode.XY_FILENAME,
        regex: str | None = None,
    ) -> ImportPlan:
        root = ensure_regular_directory(directory)
        sources = tuple(
            ImportSource(str(path), path.name, path.stat().st_size)
            for path in sorted(root.iterdir(), key=lambda item: item.name.casefold())
            if path.is_file()
        )
        return ImportPlanner().plan(
            width=project.width,
            height=project.height,
            sources=sources,
            mode=mode,
            regex=regex,
        )
