"""One layer-deletion policy, independent of storage and presentation."""

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import ContextManager, Protocol

from kraken_manager.application.dto import ArchiveLayerCommand, DeleteLayerCommand
from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.application.lifecycle import ArchiveLayerHandler, _LifecycleHandler
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import Layer, ProjectState, StructureState


@dataclass(frozen=True, slots=True)
class StagedLayerDeletion:
    token: object
    managed_files: bool


class LayerDeletionStorage(Protocol):
    def lock(self, project_id: str) -> ContextManager[None]: ...

    def is_bound(self, project_id: str, layer_id: str) -> bool: ...

    def has_running_runs(self, project_id: str, layer_id: str) -> bool: ...

    def stage(self, project_id: str, layer_id: str, delete_id: str) -> StagedLayerDeletion: ...

    def rollback(self, stage: StagedLayerDeletion) -> None: ...

    def complete(self, project_id: str, layer_id: str, stage: StagedLayerDeletion) -> None: ...


class DeleteLayerHandler(_LifecycleHandler):
    def __init__(self, uow_factory, storage_profiles, clock, storage: LayerDeletionStorage, *, archive=None):
        super().__init__(uow_factory, storage_profiles, clock)
        self._storage = storage
        self._archive = archive or ArchiveLayerHandler(uow_factory, storage_profiles, clock)

    def __call__(self, command: DeleteLayerCommand) -> Layer:
        with self._storage.lock(str(command.project_id)):
            return self._delete(command)

    def _delete(self, command: DeleteLayerCommand) -> Layer:
        delete_id = sha256(
            f"{command.project_id}:{command.layer_id}:{command.context.actor.id}:{command.context.idempotency_key}".encode()
        ).hexdigest()
        archive_context = replace(command.context, idempotency_key=f"delete-layer:{delete_id}")
        with self._uow_factory() as uow:
            project = uow.projections.get_project(command.project_id)
            layer = uow.projections.get_layer(command.layer_id)
            if project is None or layer is None or layer.project_id != project.id:
                raise NotFoundError("Layer was not found in the project")
            self._authorize(uow, project, command, Permission.MANAGE_STRUCTURE)
            if command.confirmation_name != layer.name:
                raise ValueError("Layer name confirmation does not match")
            replay = any(
                event.event_type == "LayerArchived" and event.stream_id == f"layer:{layer.id}"
                for event in uow.event_store.find_by_idempotency_key(project.id, archive_context.idempotency_key)
            )
            if not replay:
                if project.state is ProjectState.ARCHIVED or layer.state is StructureState.ARCHIVED:
                    raise ConflictError("Archived project or layer is read-only")
                if layer.revision != command.expected_revision:
                    raise ConflictError("Layer revision changed")
            if replay and not self._storage.is_bound(str(project.id), str(layer.id)):
                return layer
            if self._storage.has_running_runs(str(project.id), str(layer.id)) or any(
                job.layer_id == layer.id and job.state.value not in {"succeeded", "failed", "cancelled"}
                for job in uow.projections.list_plugin_jobs(project.id)
            ):
                raise ConflictError("Layer has an active plugin job")
        stage = self._storage.stage(str(project.id), str(layer.id), delete_id)
        try:
            archived = self._archive(
                ArchiveLayerCommand(
                    context=archive_context,
                    project_id=project.id,
                    layer_id=layer.id,
                    expected_revision=command.expected_revision,
                )
            )
        except Exception:
            self._storage.rollback(stage)
            raise
        # Metadata is committed: subsequent failures must leave files staged.
        # Repeating this command finishes auditing/cleanup without undoing the archive.
        with self._uow_factory() as uow:
            key = f"delete-layer:{delete_id}:files"
            if not uow.event_store.find_by_idempotency_key(project.id, key):
                stream = f"layer-files:{layer.id}"
                revision = uow.event_store.current_revision(stream)
                event = EventEnvelope.create(
                    stream_id=stream,
                    project_id=project.id,
                    revision=revision + 1,
                    event_type="LayerDeletedV1",
                    payload={
                        "layer_id": str(layer.id),
                        "layer_name": layer.name,
                        "delete_id": delete_id,
                        "managed_files": stage.managed_files,
                    },
                    actor=ActorSnapshot.from_principal(command.context.actor),
                    recorded_at=self._clock.now(),
                    idempotency_key=key,
                )
                uow.event_store.append(stream, expected_revision=revision, events=(event,))
                uow.commit()
        self._storage.complete(str(project.id), str(layer.id), stage)
        return archived
