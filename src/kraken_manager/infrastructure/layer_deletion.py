"""Workspace filesystem mechanics for the layer-deletion application port."""

from threading import Thread
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from kraken_manager.application.layer_deletion import StagedLayerDeletion


class WorkspaceLayerDeletionStorage:
    def __init__(self, files):
        self._files = files

    def lock(self, project_id):
        return nullcontext() if self._files is None else self._files.registry.mutation_lock(project_id)

    def is_bound(self, project_id, layer_id):
        return self._files is not None and self._files.registry.get_layer(project_id, layer_id) is not None

    def has_running_runs(self, project_id, layer_id):
        return self._files is not None and any(
            run.state.value == "running" for run in self._files.registry.list_runs(project_id, layer_id)
        )

    def stage(self, project_id, layer_id, delete_id):
        files = self._files
        workspace = None if files is None else files.registry.get_project(project_id)
        binding = None if files is None else files.registry.get_layer(project_id, layer_id)
        if workspace is None or binding is None:
            raise ValueError("Layer has no two-root file binding")
        token = files.stage_layer_deletion(project=workspace, binding=binding, delete_id=delete_id)
        return StagedLayerDeletion(token, binding.mode.value == "managed_copy")

    def rollback(self, stage):
        self._files.rollback_layer_deletion(stage.token)

    def complete(self, project_id, layer_id, stage):
        self._files.registry.remove_layer(project_id, layer_id)

        def purge():
            with self._files.registry.mutation_lock(project_id):
                current = self._files.registry.get_project(project_id)
                token = stage.token
                if current is not None:
                    token = replace(
                        token,
                        trash_roots=tuple(
                            Path(root) / "_trash" / token.delete_id
                            for root in (current.source_project_dir, current.derived_project_dir)
                        ),
                    )
                self._files.purge_layer_deletion(token)

        Thread(
            target=purge,
            name=f"kraken-layer-trash-{stage.token.delete_id}",
            daemon=True,
        ).start()
