"""composition: publish workspace plugin result."""
from __future__ import annotations
from kraken_core.plugin_protocol import (
    WorkspacePluginResultV1,
)
from kraken_manager.workspace import (
    DerivedRun,
    DerivedRunKind,
    WorkspaceValidationError,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    Layer,
    Project,
    Representation,
    RepresentationKind,
    RepresentationPurpose,
)


class PublishWorkspacePluginResultMixin:
    def publish_workspace_plugin_result(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        result: WorkspacePluginResultV1,
    ) -> tuple[DerivedRun, Representation | None]:
        run = self.workspace_registry.get_run(str(project.id), result.run_id)
        if run is None or run.layer_id != str(layer.id):
            raise WorkspaceValidationError("workspace result does not belong to this layer")
        if run.plugin_id != result.plugin_id or run.operation != result.operation:
            raise WorkspaceValidationError("workspace result provenance does not match the run")
        if result.outcome != "succeeded":
            raise WorkspaceValidationError(
                f"plugin result is not publishable: {result.outcome}"
            )
        published = self.workspace_files.publish_run(
            project_id=str(project.id),
            run_id=run.run_id,
            output_directory=result.output_directory,
            provenance=dict(result.provenance),
        )
        self._record_workspace_event(
            principal=principal,
            project_id=project.id,
            stream_id=f"derived-run:{published.run_id}",
            event_type="DerivedRunPublishedV1",
            payload={
                "run_id": published.run_id,
                "layer_id": published.layer_id,
                "kind": published.kind.value,
                "state": published.state.value,
                "path": published.path,
                "plugin_id": published.plugin_id,
                "operation": published.operation,
                "provenance": dict(published.provenance),
            },
            idempotency_key=f"derived-run:{published.run_id}:publish",
        )
        representation: Representation | None = None
        if published.kind in {DerivedRunKind.VECTOR, DerivedRunKind.RESULT}:
            current_project = self.get_project(project.id) or project
            current_layer = next(
                (item for item in self.list_layers(project.id) if item.id == layer.id),
                layer,
            )
            kind = (
                RepresentationKind.VECTOR
                if published.kind is DerivedRunKind.VECTOR
                else RepresentationKind.IMAGE
            )
            representation = self.create_representation(
                principal=principal,
                project=current_project,
                layer=current_layer,
                name=f"{layer.name} · {published.kind.value} · {published.run_id[:8]}",
                kind=kind,
                idempotency_key=f"workspace-run:{published.run_id}",
                source=published.path,
                active=True,
                purpose=(
                    RepresentationPurpose.SOURCE
                    if kind is RepresentationKind.VECTOR
                    else RepresentationPurpose.BINARY
                ),
            )
        return published, representation
