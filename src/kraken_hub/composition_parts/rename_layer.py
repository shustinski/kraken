"""composition: rename layer."""
from __future__ import annotations
import os
from dataclasses import replace
from pathlib import Path
from kraken_manager.application.dto import (
    CommandContext,
    RenameLayerCommand,
)
from kraken_manager.workspace import (
    DerivedRunKind,
    LayerSourceMode,
    WorkspaceValidationError,
    validate_workspace_name,
)
from kraken_manager.application.lifecycle import (
    RenameLayerHandler,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    Layer,
    Project,
)


class RenameLayerMixin:
    def rename_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        name: str,
        idempotency_key: str,
    ) -> Layer:
        from kraken_hub import composition as _composition

        RenameLayerHandler = _composition.RenameLayerHandler
        binding = self.layer_file_binding(project.id, layer.id)
        workspace = self.project_workspace(project.id)
        normalized = validate_workspace_name(name, field_name="Название слоя")
        if binding is None or workspace is None:
            return RenameLayerHandler(self._uow(str(project.id)), self.profiles, self.clock)(
                RenameLayerCommand(
                    context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                    project_id=project.id,
                    layer_id=layer.id,
                    name=normalized,
                    expected_revision=layer.revision,
                )
            )
        if normalized == binding.layer_name:
            return layer
        active_runs = tuple(
            run
            for run in self.list_derived_runs(project.id, layer.id)
            if run.state.value == "running"
        )
        active_jobs = tuple(
            job
            for job in self.plugin_jobs()
            if str(job.project_id) == str(project.id)
            and str(job.layer_id) == str(layer.id)
            and job.state.value not in {"succeeded", "failed", "cancelled"}
        )
        if active_runs or active_jobs:
            raise WorkspaceValidationError(
                "Нельзя переименовать слой, пока выполняются задания."
            )
        if binding.mode is LayerSourceMode.EXTERNAL:
            try:
                self.workspace_registry.save_layer(
                    str(project.id),
                    replace(binding, layer_name=normalized),
                )
                renamed = RenameLayerHandler(
                    self._uow(str(project.id)), self.profiles, self.clock
                )(
                    RenameLayerCommand(
                        context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                        project_id=project.id,
                        layer_id=layer.id,
                        name=normalized,
                        expected_revision=layer.revision,
                    )
                )
            except (OSError, RuntimeError, ValueError):
                self.workspace_registry.save_layer(str(project.id), binding)
                raise
            return renamed

        source_root = Path(workspace.source_project_dir).resolve()
        derived_root = Path(workspace.derived_project_dir).resolve()
        source_paths = tuple(
            Path(value).resolve()
            for value in (
                binding.image_directory,
                binding.ssc_directory,
                binding.prv_directory,
                binding.aux_directory,
            )
            if value
        )
        moves: list[tuple[Path, Path]] = [
            (source, source.parent / normalized)
            for source in source_paths
        ]
        moves.extend(
            (
                derived_root / kind.value / binding.layer_name,
                derived_root / kind.value / normalized,
            )
            for kind in DerivedRunKind
        )
        for source, target in moves:
            if source.exists() and target.exists():
                raise FileExistsError(f"Целевая папка слоя уже существует: {target}")
            try:
                source.relative_to(source_root if source in source_paths else derived_root)
                target.relative_to(source_root if source in source_paths else derived_root)
            except ValueError as exc:
                raise WorkspaceValidationError(
                    f"Путь слоя выходит за границы рабочего каталога: {source}"
                ) from exc
        updated_binding = replace(
            binding,
            layer_name=normalized,
            image_directory=str(Path(binding.image_directory).parent / normalized),
            ssc_directory=str(Path(binding.ssc_directory).parent / normalized),
            prv_directory=str(Path(binding.prv_directory).parent / normalized),
            aux_directory=str(Path(binding.aux_directory).parent / normalized),
        )
        runs = self.list_derived_runs(project.id, layer.id)
        updated_runs = tuple(
            replace(
                run,
                path=str(
                    Path(run.path).parent / normalized
                    if Path(run.path).name == binding.layer_name
                    else Path(run.path)
                ),
            )
            for run in runs
        )
        completed: list[tuple[Path, Path]] = []
        try:
            for source, target in moves:
                if not source.exists():
                    continue
                os.replace(source, target)
                completed.append((target, source))
            self.workspace_registry.save_layer(str(project.id), updated_binding)
            for run in updated_runs:
                self.workspace_registry.save_run(str(project.id), run)
            renamed = RenameLayerHandler(
                self._uow(str(project.id)), self.profiles, self.clock
            )(
                RenameLayerCommand(
                    context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                    project_id=project.id,
                    layer_id=layer.id,
                    name=normalized,
                    expected_revision=layer.revision,
                )
            )
        except Exception:
            for current, original in reversed(completed):
                if current.exists() and not original.exists():
                    os.replace(current, original)
            self.workspace_registry.save_layer(str(project.id), binding)
            for run in runs:
                self.workspace_registry.save_run(str(project.id), run)
            raise
        self._external_source_indexes.clear()
        return renamed
