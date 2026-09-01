"""Audited lifecycle commands for projects and layers."""

from __future__ import annotations

from collections.abc import Mapping

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    ArchiveLayerCommand,
    ArchiveProjectCommand,
    RenameLayerCommand,
    RenameProjectCommand,
    ReorderLayerCommand,
    ReorderLayersCommand,
    RestoreProjectCommand,
)
from kraken_manager.application.errors import ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.application.ports import Clock, StorageProfileCatalog, UnitOfWork, UnitOfWorkFactory
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import Layer, Project, ProjectState, StructureState


def _project_stream(project: Project) -> str:
    return f"project:{project.id}"


def _layer_stream(layer: Layer) -> str:
    return f"layer:{layer.id}"


def _project_payload(project: Project) -> dict[str, object]:
    return {
        "id": str(project.id),
        "name": project.name,
        "width": project.width,
        "height": project.height,
        "orientation": project.orientation.value,
        "storage_profile": project.storage_profile,
        "state": project.state.value,
        "revision": project.revision,
        "created_at": project.created_at.isoformat(),
    }


def _layer_payload(layer: Layer) -> dict[str, object]:
    return {
        "id": str(layer.id),
        "project_id": str(layer.project_id),
        "name": layer.name,
        "type": layer.type.value,
        "order": layer.order,
        "state": layer.state.value,
        "revision": layer.revision,
        "created_at": layer.created_at.isoformat(),
    }


class _LifecycleHandler:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        storage_profiles: StorageProfileCatalog,
        clock: Clock,
        authorization: AuthorizationPolicy | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._profiles = storage_profiles
        self._clock = clock
        self._authorization = authorization or AuthorizationPolicy()

    def _authorize(
        self,
        uow: UnitOfWork,
        project: Project,
        command: object,
        permission: Permission,
    ) -> None:
        profile = self._profiles.get(project.storage_profile)
        if profile is None:
            raise NotFoundError(f"Storage profile {project.storage_profile!r} was not found")
        context = command.context
        self._authorization.decide(
            principal=context.actor,
            storage=profile,
            permission=permission,
            roles=uow.acl.roles_for(project.id, context.actor.id),
            gitlab_identity_verified=context.gitlab_identity_verified,
        ).require()

    @staticmethod
    def _was_applied(uow: UnitOfWork, project: Project, key: str, event_type: str) -> bool:
        return any(
            event.event_type == event_type
            for event in uow.event_store.find_by_idempotency_key(project.id, key)
        )

    def _event(
        self,
        *,
        command: object,
        project: Project,
        stream_id: str,
        revision: int,
        event_type: str,
        payload: Mapping[str, object],
    ) -> EventEnvelope:
        context = command.context
        return EventEnvelope.create(
            stream_id=stream_id,
            project_id=project.id,
            revision=revision,
            event_type=event_type,
            payload=payload,
            actor=ActorSnapshot.from_principal(context.actor),
            recorded_at=self._clock.now(),
            effective_at=context.effective_at,
            performer_id=context.performer_id,
            correlation_id=context.correlation_id,
            idempotency_key=context.idempotency_key,
        )


class _ProjectMutationHandler(_LifecycleHandler):
    event_type: str
    permission: Permission

    def mutate(self, project: Project, command: object) -> Project:
        raise NotImplementedError

    def __call__(self, command: object) -> Project:
        with self._uow_factory() as uow:
            project = uow.projections.get_project(command.project_id)
            if project is None:
                raise NotFoundError(f"Project {command.project_id} was not found")
            if self._was_applied(uow, project, command.context.idempotency_key, self.event_type):
                return project
            self._authorize(uow, project, command, self.permission)
            if project.revision != command.expected_revision:
                raise ConcurrencyError(
                    f"Expected project revision {command.expected_revision}, found {project.revision}"
                )
            next_project = self.mutate(project, command)
            event = self._event(
                command=command,
                project=project,
                stream_id=_project_stream(project),
                revision=next_project.revision,
                event_type=self.event_type,
                payload={"project": _project_payload(next_project)},
            )
            uow.event_store.append(
                _project_stream(project), expected_revision=project.revision, events=(event,)
            )
            uow.projections.save_project(next_project)
            uow.commit()
            return next_project


class RenameProjectHandler(_ProjectMutationHandler):
    event_type = "ProjectRenamed"
    permission = Permission.RENAME_PROJECT

    def mutate(self, project: Project, command: RenameProjectCommand) -> Project:
        if project.state is ProjectState.ARCHIVED:
            raise ConflictError("Archived project is read-only")
        return project.rename(command.name, expected_revision=command.expected_revision)


class ArchiveProjectHandler(_ProjectMutationHandler):
    event_type = "ProjectArchived"
    permission = Permission.ARCHIVE_PROJECT

    def mutate(self, project: Project, command: ArchiveProjectCommand) -> Project:
        if project.state is ProjectState.ARCHIVED:
            raise ConflictError("Project is already archived")
        return project.archive(expected_revision=command.expected_revision)


class RestoreProjectHandler(_ProjectMutationHandler):
    event_type = "ProjectRestored"
    permission = Permission.ARCHIVE_PROJECT

    def mutate(self, project: Project, command: RestoreProjectCommand) -> Project:
        if project.state is ProjectState.ACTIVE:
            raise ConflictError("Project is already active")
        return project.restore(expected_revision=command.expected_revision)


class _LayerMutationHandler(_LifecycleHandler):
    event_type: str

    def mutate(self, layer: Layer, command: object) -> Layer:
        raise NotImplementedError

    def __call__(self, command: object) -> Layer:
        with self._uow_factory() as uow:
            project = uow.projections.get_project(command.project_id)
            if project is None:
                raise NotFoundError(f"Project {command.project_id} was not found")
            layer = uow.projections.get_layer(command.layer_id)
            if layer is None or layer.project_id != project.id:
                raise NotFoundError("Layer was not found in the project")
            if self._was_applied(uow, project, command.context.idempotency_key, self.event_type):
                return layer
            self._authorize(uow, project, command, Permission.MANAGE_STRUCTURE)
            if project.state is ProjectState.ARCHIVED:
                raise ConflictError("Archived project is read-only")
            if layer.revision != command.expected_revision:
                raise ConcurrencyError(
                    f"Expected layer revision {command.expected_revision}, found {layer.revision}"
                )
            next_layer = self.mutate(layer, command)
            event = self._event(
                command=command,
                project=project,
                stream_id=_layer_stream(layer),
                revision=next_layer.revision,
                event_type=self.event_type,
                payload={"layer": _layer_payload(next_layer)},
            )
            uow.event_store.append(
                _layer_stream(layer), expected_revision=layer.revision, events=(event,)
            )
            uow.projections.save_layer(next_layer)
            uow.commit()
            return next_layer


class RenameLayerHandler(_LayerMutationHandler):
    event_type = "LayerRenamed"

    def mutate(self, layer: Layer, command: RenameLayerCommand) -> Layer:
        if layer.state is StructureState.ARCHIVED:
            raise ConflictError("Archived layer is read-only")
        return layer.rename(command.name, expected_revision=command.expected_revision)


class ReorderLayerHandler(_LayerMutationHandler):
    event_type = "LayerReordered"

    def mutate(self, layer: Layer, command: ReorderLayerCommand) -> Layer:
        if layer.state is StructureState.ARCHIVED:
            raise ConflictError("Archived layer is read-only")
        return layer.reorder(command.order, expected_revision=command.expected_revision)


class ReorderLayersHandler(_LifecycleHandler):
    """Persist a complete normalized order in one unit-of-work transaction."""

    event_type = "LayersReordered"

    def __call__(self, command: ReorderLayersCommand) -> tuple[Layer, ...]:
        with self._uow_factory() as uow:
            project = uow.projections.get_project(command.project_id)
            if project is None:
                raise NotFoundError(f"Project {command.project_id} was not found")
            current = tuple(uow.projections.list_layers(project.id))
            if self._was_applied(uow, project, command.context.idempotency_key, self.event_type):
                by_id = {layer.id: layer for layer in current}
                return tuple(by_id[identifier] for identifier in command.layer_ids)
            self._authorize(uow, project, command, Permission.MANAGE_STRUCTURE)
            if project.state is ProjectState.ARCHIVED:
                raise ConflictError("Archived project is read-only")
            by_id = {layer.id: layer for layer in current}
            if set(by_id) != set(command.layer_ids):
                raise ConflictError("Layer order must contain every active project layer exactly once")
            expected = dict(command.expected_revisions)
            for identifier, layer in by_id.items():
                if layer.revision != expected[identifier]:
                    raise ConcurrencyError(
                        f"Expected layer revision {expected[identifier]}, found {layer.revision}"
                    )

            reordered = tuple(
                by_id[identifier].reorder(order, expected_revision=expected[identifier])
                for order, identifier in enumerate(command.layer_ids)
            )
            events_by_stream: list[tuple[str, int, EventEnvelope]] = []
            order_payload = [str(identifier) for identifier in command.layer_ids]
            for layer in reordered:
                stream = _layer_stream(by_id[layer.id])
                event = self._event(
                    command=command,
                    project=project,
                    stream_id=stream,
                    revision=layer.revision,
                    event_type=self.event_type,
                    payload={"layer": _layer_payload(layer), "layer_ids": order_payload},
                )
                events_by_stream.append((stream, by_id[layer.id].revision, event))
            for stream, revision, event in events_by_stream:
                uow.event_store.append(stream, expected_revision=revision, events=(event,))
            for layer in reordered:
                uow.projections.save_layer(layer)
            uow.commit()
            return reordered


class ArchiveLayerHandler(_LayerMutationHandler):
    event_type = "LayerArchived"

    def mutate(self, layer: Layer, command: ArchiveLayerCommand) -> Layer:
        if layer.state is StructureState.ARCHIVED:
            raise ConflictError("Layer is already archived")
        return layer.archive(expected_revision=command.expected_revision)


__all__ = [
    "ArchiveLayerHandler",
    "ArchiveProjectHandler",
    "RenameLayerHandler",
    "RenameProjectHandler",
    "ReorderLayerHandler",
    "ReorderLayersHandler",
    "RestoreProjectHandler",
]
