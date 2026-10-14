"""composition: rename project."""
from __future__ import annotations
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from kraken_manager.application.dto import (
    CommandContext,
    RenameProjectCommand,
)
from kraken_manager.workspace import (
    WorkspaceValidationError,
    validate_workspace_name,
)
from kraken_manager.application.lifecycle import (
    RenameProjectHandler,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    Project,
)


class RenameProjectMixin:
    def rename_project(
        self, *, principal: Principal, project: Project, name: str, idempotency_key: str
    ) -> Project:
        from kraken_hub import composition as _composition

        RenameProjectHandler = _composition.RenameProjectHandler
        workspace = self.project_workspace(project.id)
        if workspace is None:
            return RenameProjectHandler(self._uow(str(project.id)), self.profiles, self.clock)(
                RenameProjectCommand(
                    context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                    project_id=project.id,
                    name=name,
                    expected_revision=project.revision,
                )
            )
        normalized = validate_workspace_name(name, field_name="Название проекта")
        if normalized == workspace.project_name:
            return project
        active_runs = tuple(
            run for run in self.list_derived_runs(project.id) if run.state.value == "running"
        )
        latest_job_states: dict[str, str] = {}
        for event in self.history(project.id):
            payload = getattr(event, "payload", {})
            job = payload.get("job", {}) if isinstance(payload, Mapping) else {}
            if not isinstance(job, Mapping):
                continue
            job_id = str(payload.get("plugin_job_id", job.get("id", "")))
            if job_id:
                latest_job_states[job_id] = str(job.get("state", ""))
        if active_runs or any(
            state and state not in {"succeeded", "failed", "cancelled"}
            for state in latest_job_states.values()
        ):
            raise WorkspaceValidationError(
                "Нельзя переименовать проект, пока выполняются задания."
            )

        source_old = Path(workspace.source_project_dir).resolve()
        derived_old = Path(workspace.derived_project_dir).resolve()
        source_new = Path(workspace.source_root).resolve() / normalized
        derived_new = Path(workspace.derived_root).resolve() / normalized
        if source_new.exists() or derived_new.exists():
            raise FileExistsError(
                f"Целевая папка проекта «{normalized}» уже существует."
            )
        layer_bindings = tuple(
            binding
            for layer in self.list_layers(project.id, include_archived=True)
            if (binding := self.layer_file_binding(project.id, layer.id)) is not None
        )
        runs = self.list_derived_runs(project.id)

        def remap(value: str, old_root: Path, new_root: Path) -> str:
            if not value:
                return value
            candidate = Path(value).resolve(strict=False)
            try:
                relative = candidate.relative_to(old_root)
            except ValueError:
                return value
            return str(new_root / relative)

        updated_workspace = replace(
            workspace,
            project_name=normalized,
            source_project_dir=str(source_new),
            derived_project_dir=str(derived_new),
        )
        updated_layers = tuple(
            replace(
                binding,
                image_directory=remap(binding.image_directory, source_old, source_new),
                ssc_directory=remap(binding.ssc_directory, source_old, source_new),
                prv_directory=remap(binding.prv_directory, source_old, source_new),
                aux_directory=remap(binding.aux_directory, source_old, source_new),
                import_root=remap(binding.import_root, source_old, source_new),
            )
            for binding in layer_bindings
        )
        updated_runs = tuple(
            replace(run, path=remap(run.path, derived_old, derived_new))
            for run in runs
        )
        moved: list[tuple[Path, Path]] = []
        try:
            os.replace(source_old, source_new)
            moved.append((source_new, source_old))
            os.replace(derived_old, derived_new)
            moved.append((derived_new, derived_old))
            self.workspace_registry.save_project(updated_workspace)
            for binding in updated_layers:
                self.workspace_registry.save_layer(str(project.id), binding)
            for run in updated_runs:
                self.workspace_registry.save_run(str(project.id), run)
            renamed = RenameProjectHandler(
                self._uow(str(project.id)), self.profiles, self.clock
            )(
                RenameProjectCommand(
                    context=CommandContext(
                        actor=principal,
                        idempotency_key=idempotency_key,
                    ),
                    project_id=project.id,
                    name=normalized,
                    expected_revision=project.revision,
                )
            )
        except Exception:
            for current, original in reversed(moved):
                if current.exists() and not original.exists():
                    os.replace(current, original)
            self.workspace_registry.save_project(workspace)
            for binding in layer_bindings:
                self.workspace_registry.save_layer(str(project.id), binding)
            for run in runs:
                self.workspace_registry.save_run(str(project.id), run)
            raise
        self._external_source_indexes.clear()
        return renamed
