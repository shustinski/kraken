"""Rename policies shared by all storage adapters."""

from dataclasses import replace
from typing import ContextManager, Protocol

from kraken_manager.application.errors import ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.application.lifecycle import _LifecycleHandler
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import ProjectState, StructureState


class WorkspaceRenameStorage(Protocol):
    def lock(self, project_id: str) -> ContextManager[None]: ...

    def has_running_runs(self, project_id: str, layer_id: str = "") -> bool: ...

    def stage_project(self, project_id: str, name: str) -> object: ...

    def stage_layer(self, project_id: str, layer_id: str, name: str) -> object: ...

    def rollback(self, token: object) -> None: ...

    def complete(self, token: object) -> None: ...

    def reconcile(self, project_id: str) -> None: ...


class RenameWorkspaceHandler(_LifecycleHandler):
    def __init__(self, uow_factory, profiles, clock, storage: WorkspaceRenameStorage, metadata_handler, *, layer: bool):
        super().__init__(uow_factory, profiles, clock)
        self._storage = storage
        self._metadata_handler = metadata_handler
        self._layer = layer

    def __call__(self, command):
        with self._storage.lock(str(command.project_id)):
            return self._rename(command)

    def _rename(self, command):
        name = command.name.strip()
        layer_id = str(command.layer_id) if self._layer else ""
        with self._uow_factory() as unit:
            project = unit.projections.get_project(command.project_id)
            if project is None:
                raise NotFoundError("Project was not found")
            entity = unit.projections.get_layer(command.layer_id) if self._layer else project
            if entity is None or (self._layer and entity.project_id != project.id):
                raise NotFoundError("Layer was not found in the project")
            self._authorize(
                unit, project, command, Permission.MANAGE_STRUCTURE if self._layer else Permission.RENAME_PROJECT
            )
            event_type = "LayerRenamed" if self._layer else "ProjectRenamed"
            if self._was_applied(unit, project, command.context.idempotency_key, event_type):
                self._storage.reconcile(str(project.id))
                return entity
            if project.state is ProjectState.ARCHIVED or (self._layer and entity.state is StructureState.ARCHIVED):
                raise ConflictError("Archived project or layer is read-only")
            if entity.revision != command.expected_revision:
                raise ConcurrencyError(f"Expected revision {command.expected_revision}, found {entity.revision}")
            if entity.name == name:
                return entity
            if self._storage.has_running_runs(str(project.id), layer_id) or any(
                (not self._layer or str(job.layer_id) == layer_id)
                and job.state.value not in {"succeeded", "failed", "cancelled"}
                for job in unit.projections.list_plugin_jobs(project.id)
            ):
                raise ConflictError("Нельзя переименовать объект, пока выполняются задания.")
        token = (
            self._storage.stage_layer(str(project.id), layer_id, name)
            if self._layer
            else self._storage.stage_project(str(project.id), name)
        )
        try:
            result = self._metadata_handler(replace(command, name=name))
        except Exception:
            self._storage.rollback(token)
            raise
        self._storage.complete(token)
        return result
