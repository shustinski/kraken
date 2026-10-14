"""composition: delete layer."""
from __future__ import annotations
from threading import Thread
from kraken_manager.workspace import (
    WorkspaceValidationError,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    Layer,
    Project,
)


class DeleteLayerMixin:
    def delete_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        confirmation_name: str,
        idempotency_key: str,
    ) -> Layer:
        if confirmation_name != layer.name:
            raise WorkspaceValidationError("layer name confirmation does not match")
        workspace = self.project_workspace(project.id)
        binding = self.layer_file_binding(project.id, layer.id)
        if workspace is None or binding is None:
            raise WorkspaceValidationError("layer has no two-root file binding")
        active_runs = [
            run
            for run in self.workspace_registry.list_runs(str(project.id), str(layer.id))
            if run.state.value == "running"
        ]
        active_jobs = [
            job
            for job in self.plugin_jobs()
            if str(job.project_id) == str(project.id)
            and str(job.layer_id) == str(layer.id)
            and job.state.value not in {"succeeded", "failed", "cancelled"}
        ]
        if active_runs or active_jobs:
            raise WorkspaceValidationError("layer has an active plugin job")
        stage = self.workspace_files.stage_layer_deletion(
            project=workspace,
            binding=binding,
            delete_id=idempotency_key,
        )
        try:
            current_project = self.get_project(project.id) or project
            current_layer = next(
                (item for item in self.list_layers(project.id) if item.id == layer.id),
                None,
            )
            if current_layer is None:
                raise WorkspaceValidationError("layer is no longer available")
            archived = self.archive_layer(
                principal=principal,
                project=current_project,
                layer=current_layer,
                idempotency_key=idempotency_key,
            )
            self._record_workspace_event(
                principal=principal,
                project_id=project.id,
                stream_id=f"layer-files:{layer.id}",
                event_type="LayerDeletedV1",
                payload={
                    "layer_id": str(layer.id),
                    "layer_name": layer.name,
                    "delete_id": idempotency_key,
                    "managed_files": binding.mode.value == "managed_copy",
                },
                idempotency_key=f"{idempotency_key}:files",
            )
            self.workspace_registry.remove_layer(str(project.id), str(layer.id))
        except Exception:
            self.workspace_files.rollback_layer_deletion(stage)
            raise
        Thread(
            target=self.workspace_files.purge_layer_deletion,
            args=(stage,),
            name=f"kraken-layer-trash-{idempotency_key}",
            daemon=True,
        ).start()
        return archived
