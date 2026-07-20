"""Audited representation changes serialized by their parent layer stream."""

from __future__ import annotations

from dataclasses import replace

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    ActivateRepresentationCommand,
    ArchiveRepresentationCommand,
    RenameRepresentationCommand,
    UpdateRepresentationNoteCommand,
)
from kraken_manager.application.errors import ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.application.ports import Clock, StorageProfileCatalog, UnitOfWorkFactory
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import ProjectState, Representation, StructureState


def _snapshot(value: Representation) -> dict[str, object]:
    return {
        "id": str(value.id),
        "project_id": str(value.project_id),
        "layer_id": str(value.layer_id),
        "name": value.name,
        "kind": value.kind.value,
        "note": value.note,
        "source": value.source,
        "active": value.active,
        "state": value.state.value,
        "revision": value.revision,
        "created_at": value.created_at.isoformat(),
    }


class _RepresentationHandler:
    event_type: str

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

    def mutate(self, value: Representation, command: object) -> Representation:
        raise NotImplementedError

    def __call__(self, command: object) -> Representation:
        with self._uow_factory() as uow:
            project = uow.projections.get_project(command.project_id)
            layer = uow.projections.get_layer(command.layer_id)
            value = uow.projections.get_representation(command.representation_id)
            if project is None:
                raise NotFoundError("Project was not found")
            if layer is None or layer.project_id != project.id:
                raise NotFoundError("Layer was not found in the project")
            if value is None or value.layer_id != layer.id:
                raise NotFoundError("Representation was not found in the layer")
            if any(
                event.event_type == self.event_type
                for event in uow.event_store.find_by_idempotency_key(
                    project.id, command.context.idempotency_key
                )
            ):
                return value
            profile = self._profiles.get(project.storage_profile)
            if profile is None:
                raise NotFoundError("Project storage profile was not found")
            self._authorization.decide(
                principal=command.context.actor,
                storage=profile,
                permission=Permission.MANAGE_STRUCTURE,
                roles=uow.acl.roles_for(project.id, command.context.actor.id),
                gitlab_identity_verified=command.context.gitlab_identity_verified,
            ).require()
            if project.state is ProjectState.ARCHIVED or layer.state is StructureState.ARCHIVED:
                raise ConflictError("Archived structure is read-only")
            if layer.revision != command.expected_layer_revision:
                raise ConcurrencyError(
                    f"Expected layer revision {command.expected_layer_revision}, found {layer.revision}"
                )
            if value.revision != command.expected_representation_revision:
                raise ConcurrencyError(
                    "Representation revision changed"
                )
            next_value = self.mutate(value, command)
            deactivated: list[Representation] = []
            if self.event_type == "RepresentationActivated":
                for previous in uow.projections.list_representations(layer.id):
                    if previous.id != value.id and previous.kind is value.kind and previous.active:
                        deactivated.append(previous.deactivate())
            next_layer = replace(layer, revision=layer.revision + 1)
            now = self._clock.now()
            event = EventEnvelope.create(
                stream_id=f"layer:{layer.id}",
                project_id=project.id,
                revision=next_layer.revision,
                event_type=self.event_type,
                payload={
                    "representation": _snapshot(next_value),
                    "deactivated": [_snapshot(item) for item in deactivated],
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=command.context.performer_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(
                f"layer:{layer.id}", expected_revision=layer.revision, events=(event,)
            )
            for previous in deactivated:
                uow.projections.save_representation(previous)
            uow.projections.save_representation(next_value)
            uow.projections.save_layer(next_layer)
            uow.commit()
            return next_value


class RenameRepresentationHandler(_RepresentationHandler):
    event_type = "RepresentationRenamed"

    def mutate(self, value: Representation, command: RenameRepresentationCommand) -> Representation:
        if value.state is StructureState.ARCHIVED:
            raise ConflictError("Archived representation is read-only")
        return value.rename(command.name, expected_revision=command.expected_representation_revision)


class UpdateRepresentationNoteHandler(_RepresentationHandler):
    event_type = "RepresentationNoteUpdated"

    def mutate(self, value: Representation, command: UpdateRepresentationNoteCommand) -> Representation:
        if value.state is StructureState.ARCHIVED:
            raise ConflictError("Archived representation is read-only")
        return value.update_note(command.note, expected_revision=command.expected_representation_revision)


class ActivateRepresentationHandler(_RepresentationHandler):
    event_type = "RepresentationActivated"

    def mutate(self, value: Representation, command: ActivateRepresentationCommand) -> Representation:
        if value.state is StructureState.ARCHIVED:
            raise ConflictError("Archived representation cannot be activated")
        if value.active:
            raise ConflictError("Representation is already active")
        return value.activate(expected_revision=command.expected_representation_revision)


class ArchiveRepresentationHandler(_RepresentationHandler):
    event_type = "RepresentationArchived"

    def mutate(self, value: Representation, command: ArchiveRepresentationCommand) -> Representation:
        if value.state is StructureState.ARCHIVED:
            raise ConflictError("Representation is already archived")
        return value.archive(expected_revision=command.expected_representation_revision)


__all__ = [
    "ActivateRepresentationHandler",
    "ArchiveRepresentationHandler",
    "RenameRepresentationHandler",
    "UpdateRepresentationNoteHandler",
]
