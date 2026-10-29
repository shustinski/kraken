"""Stage workspace path changes and restore them if metadata commit fails."""

import os
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path

from kraken_core.safe_files import is_link_or_reparse
from kraken_manager.workspace import DerivedRunKind, LayerSourceMode, validate_workspace_name


@dataclass
class WorkspaceRenameStage:
    project_id: str
    workspace: object = None
    layers: tuple = ()
    runs: tuple = ()
    moves: list = field(default_factory=list)
    changed_workspace: object = None


class FilesystemWorkspaceRenameStorage:
    def __init__(self, files, *, invalidate=lambda: None, project_changed=None):
        self._files = files
        self._invalidate = invalidate
        self._project_changed = project_changed

    def lock(self, project_id):
        return nullcontext() if self._files is None else self._files.registry.mutation_lock(project_id)

    def has_running_runs(self, project_id, layer_id=""):
        return self._files is not None and any(
            run.state.value == "running" for run in self._files.registry.list_runs(project_id, layer_id)
        )

    @staticmethod
    def _remap(value, moves):
        if not value:
            return value
        candidate = Path(value).resolve()
        for old, new in moves:
            if candidate.is_relative_to(old):
                return str(new / candidate.relative_to(old))
        return value

    @staticmethod
    def _move(stage, moves):
        # A server may place source and derived trees in the same root.
        for old, new in dict(moves).items():
            if old == new or not old.exists():
                continue
            if new.exists():
                raise FileExistsError(f"Workspace destination already exists: {new}")
            os.replace(old, new)
            stage.moves.append((old, new))

    def stage_project(self, project_id, name):
        name = validate_workspace_name(name, field_name="Название проекта")
        registry = None if self._files is None else self._files.registry
        workspace = None if registry is None else registry.get_project(project_id)
        stage = WorkspaceRenameStage(project_id)
        if workspace is None:
            return stage
        old_source, old_derived = (
            Path(workspace.source_project_dir).resolve(),
            Path(workspace.derived_project_dir).resolve(),
        )
        source_root, derived_root = Path(workspace.source_root).resolve(), Path(workspace.derived_root).resolve()
        for old, root, original in (
            (old_source, source_root, workspace.source_project_dir),
            (old_derived, derived_root, workspace.derived_project_dir),
        ):
            if old == root or not old.is_relative_to(root) or is_link_or_reparse(Path(original)):
                raise ValueError("Project path is outside its workspace root or is a link")
            if not old.is_dir():
                raise FileNotFoundError(old)
        new_source, new_derived = source_root / name, derived_root / name
        moves = ((old_source, new_source), (old_derived, new_derived))
        stage.workspace = workspace
        stage.layers = registry.list_layers(project_id)
        stage.runs = registry.list_runs(project_id)
        updated = replace(
            workspace, project_name=name, source_project_dir=str(new_source), derived_project_dir=str(new_derived)
        )
        try:
            self._move(stage, moves)
            registry.save_project(updated)
            for binding in stage.layers:
                registry.save_layer(
                    project_id,
                    replace(
                        binding,
                        **{
                            field: self._remap(getattr(binding, field), moves)
                            for field in (
                                "image_directory",
                                "ssc_directory",
                                "prv_directory",
                                "aux_directory",
                                "import_root",
                            )
                        },
                    ),
                )
            for run in stage.runs:
                registry.save_run(project_id, replace(run, path=self._remap(run.path, moves)))
            stage.changed_workspace = updated
            return stage
        except Exception:
            self.rollback(stage)
            raise

    def stage_layer(self, project_id, layer_id, name):
        name = validate_workspace_name(name, field_name="Название слоя")
        registry = None if self._files is None else self._files.registry
        binding = None if registry is None else registry.get_layer(project_id, layer_id)
        workspace = None if registry is None else registry.get_project(project_id)
        stage = WorkspaceRenameStage(project_id)
        if binding is None or workspace is None:
            return stage
        stage.layers = (binding,)
        stage.runs = registry.list_runs(project_id, layer_id)
        moves = []
        updated = replace(binding, layer_name=name)
        if binding.mode is LayerSourceMode.MANAGED_COPY:
            roots = (Path(workspace.source_project_dir).resolve(), Path(workspace.derived_project_dir).resolve())
            changes = {}
            for field_name in ("image_directory", "ssc_directory", "prv_directory", "aux_directory"):
                value = getattr(binding, field_name)
                if not value:
                    continue
                old = Path(value).resolve()
                new = old.parent / name
                if not any(old.is_relative_to(root) and new.is_relative_to(root) and old != root for root in roots):
                    raise ValueError("Layer path is outside the project workspace")
                moves.append((old, new))
                changes[field_name] = str(new)
            moves.extend(
                (roots[1] / kind.value / binding.layer_name, roots[1] / kind.value / name) for kind in DerivedRunKind
            )
            updated = replace(updated, **changes)
        try:
            self._move(stage, moves)
            registry.save_layer(project_id, updated)
            for run in stage.runs:
                registry.save_run(project_id, replace(run, path=self._remap(run.path, moves)))
            return stage
        except Exception:
            self.rollback(stage)
            raise

    def rollback(self, stage):
        if self._files is None:
            return
        for old, new in reversed(stage.moves):
            if new.exists() and not old.exists():
                os.replace(new, old)
        registry = self._files.registry
        if stage.workspace is not None:
            registry.save_project(stage.workspace)
        for binding in stage.layers:
            registry.save_layer(stage.project_id, binding)
        for run in stage.runs:
            registry.save_run(stage.project_id, run)

    def complete(self, stage):
        if stage.changed_workspace is not None and self._project_changed is not None:
            self._project_changed(stage.project_id, stage.changed_workspace)
        self._invalidate()

    def reconcile(self, project_id):
        current = None if self._files is None else self._files.registry.get_project(project_id)
        if current is not None and self._project_changed is not None:
            self._project_changed(project_id, current)
        self._invalidate()
