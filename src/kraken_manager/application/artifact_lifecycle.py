"""Audited lifecycle operations for artifact series, versions, and notes."""

from __future__ import annotations

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    ActivateArtifactVersionCommand,
    AddExternalArtifactVersionCommand,
    ArchiveArtifactSeriesCommand,
    CreateNoteCommand,
    RenameArtifactSeriesCommand,
    ReviseNoteCommand,
)
from kraken_manager.application.errors import ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.application.ports import Clock, StorageProfileCatalog, UnitOfWork, UnitOfWorkFactory
from kraken_manager.domain.artifacts import (
    ArtifactSeries,
    ArtifactVersion,
    ExternalReference,
    NoteRevision,
)
from kraken_manager.domain.common import ArtifactVersionId, require_non_empty
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission


def _series_stream(series_id: object) -> str:
    return f"artifact-series:{series_id}"


def _note_stream(note_id: str) -> str:
    return f"note:{note_id}"


class _AuthorizedHandler:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        profiles: StorageProfileCatalog,
        clock: Clock,
        authorization: AuthorizationPolicy | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._profiles = profiles
        self._clock = clock
        self._authorization = authorization or AuthorizationPolicy()

    def _project(self, uow: UnitOfWork, command: object, permission: Permission):
        project = uow.projections.get_project(command.project_id)
        if project is None:
            raise NotFoundError("Project was not found")
        profile = self._profiles.get(project.storage_profile)
        if profile is None:
            raise NotFoundError("Storage profile was not found")
        self._authorization.decide(
            principal=command.context.actor,
            storage=profile,
            permission=permission,
            roles=uow.acl.roles_for(project.id, command.context.actor.id),
            gitlab_identity_verified=command.context.gitlab_identity_verified,
        ).require()
        return project

    @staticmethod
    def _prior(uow: UnitOfWork, command: object, event_type: str) -> EventEnvelope | None:
        events = uow.event_store.find_by_idempotency_key(
            command.project_id,
            command.context.idempotency_key,
        )
        matches = tuple(event for event in events if event.event_type == event_type)
        if matches:
            return matches[-1]
        if events:
            raise ConflictError("Idempotency key was already used by another command")
        return None


class RenameArtifactSeriesHandler(_AuthorizedHandler):
    def __call__(self, command: RenameArtifactSeriesCommand) -> ArtifactSeries:
        with self._uow_factory() as uow:
            self._project(uow, command, Permission.IMPORT_ARTIFACT)
            prior = self._prior(uow, command, "ArtifactSeriesRenamed")
            series = uow.projections.get_artifact_series(command.series_id)
            if series is None or series.project_id != command.project_id:
                raise NotFoundError("Artifact series was not found")
            if prior is not None:
                return series
            renamed = series.rename(
                command.name,
                expected_revision=command.expected_series_revision,
            )
            stream = _series_stream(series.id)
            stream_revision = uow.event_store.current_revision(stream)
            now = self._clock.now()
            event = EventEnvelope.create(
                stream_id=stream,
                project_id=series.project_id,
                revision=stream_revision + 1,
                event_type="ArtifactSeriesRenamed",
                payload={
                    "artifact_series_id": series.id,
                    "name": renamed.name,
                    "series_revision": renamed.revision,
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=command.context.performer_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(stream, expected_revision=stream_revision, events=(event,))
            uow.projections.save_artifact_series(renamed)
            uow.commit()
            return renamed


class ArchiveArtifactSeriesHandler(_AuthorizedHandler):
    def __call__(self, command: ArchiveArtifactSeriesCommand) -> ArtifactSeries:
        with self._uow_factory() as uow:
            self._project(uow, command, Permission.IMPORT_ARTIFACT)
            prior = self._prior(uow, command, "ArtifactSeriesArchived")
            series = uow.projections.get_artifact_series(command.series_id)
            if series is None or series.project_id != command.project_id:
                raise NotFoundError("Artifact series was not found")
            if prior is not None:
                return series
            archived = series.archive(
                expected_revision=command.expected_series_revision,
            )
            if archived is series:
                raise ConflictError("Artifact series is already archived")
            stream = _series_stream(series.id)
            stream_revision = uow.event_store.current_revision(stream)
            now = self._clock.now()
            event = EventEnvelope.create(
                stream_id=stream,
                project_id=series.project_id,
                revision=stream_revision + 1,
                event_type="ArtifactSeriesArchived",
                payload={
                    "artifact_series_id": series.id,
                    "archived": True,
                    "series_revision": archived.revision,
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=command.context.performer_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(stream, expected_revision=stream_revision, events=(event,))
            uow.projections.save_artifact_series(archived)
            uow.commit()
            return archived


class ActivateArtifactVersionHandler(_AuthorizedHandler):
    def __call__(self, command: ActivateArtifactVersionCommand) -> ArtifactVersion:
        with self._uow_factory() as uow:
            self._project(uow, command, Permission.IMPORT_ARTIFACT)
            prior = self._prior(uow, command, "ArtifactVersionActivated")
            version = uow.projections.get_artifact_version(command.version_id)
            if version is None or version.series_id != command.series_id:
                raise NotFoundError("Artifact version was not found in the series")
            if prior is not None:
                return version
            active = uow.projections.get_active_artifact_version(command.series_id)
            if active is not None and active.id == version.id:
                raise ConflictError("Artifact version is already active")
            stream = _series_stream(command.series_id)
            stream_revision = uow.event_store.current_revision(stream)
            now = self._clock.now()
            event = EventEnvelope.create(
                stream_id=stream,
                project_id=command.project_id,
                revision=stream_revision + 1,
                event_type="ArtifactVersionActivated",
                payload={
                    "artifact_version_id": version.id,
                    "series_id": version.series_id,
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=command.context.performer_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(stream, expected_revision=stream_revision, events=(event,))
            uow.projections.save_artifact_version(version, activate=True)
            uow.commit()
            return version


class AddExternalArtifactVersionHandler(_AuthorizedHandler):
    def __call__(self, command: AddExternalArtifactVersionCommand) -> ArtifactVersion:
        with self._uow_factory() as uow:
            self._project(uow, command, Permission.IMPORT_ARTIFACT)
            prior = self._prior(uow, command, "ExternalArtifactVersionAdded")
            series = uow.projections.get_artifact_series(command.series_id)
            if series is None or series.project_id != command.project_id:
                raise NotFoundError("Artifact series was not found")
            if prior is not None:
                identifier = ArtifactVersionId(str(prior.payload["artifact_version_id"]))
                existing = uow.projections.get_artifact_version(identifier)
                if existing is not None:
                    return existing
            stream = _series_stream(series.id)
            stream_revision = uow.event_store.current_revision(stream)
            if stream_revision != command.expected_series_revision:
                raise ConcurrencyError(
                    f"Expected artifact stream revision {command.expected_series_revision}, "
                    f"found {stream_revision}"
                )
            active = uow.projections.get_active_artifact_version(series.id)
            parent_id = command.parent_version_id or (active.id if active is not None else None)
            if parent_id is not None:
                parent = uow.projections.get_artifact_version(parent_id)
                if parent is None or parent.series_id != series.id:
                    raise ConflictError("Parent version does not belong to the series")
            now = self._clock.now()
            reference = ExternalReference(
                uri=command.uri,
                fingerprint_sha256=command.fingerprint_sha256,
                observed_size_bytes=command.observed_size_bytes,
            )
            version = ArtifactVersion.external_link(
                series_id=series.id,
                reference=reference,
                media_type=command.media_type,
                filename=command.filename,
                author_principal_id=command.context.actor.id,
                created_at=now,
                parent_version_id=parent_id,
                parameters=command.parameters,
            )
            activate = active is None or parent_id == active.id
            event = EventEnvelope.create(
                stream_id=stream,
                project_id=series.project_id,
                revision=stream_revision + 1,
                event_type="ExternalArtifactVersionAdded",
                payload={
                    "artifact_version_id": version.id,
                    "series_id": series.id,
                    "sha256": version.sha256,
                    "size_bytes": version.size_bytes,
                    "media_type": version.media_type,
                    "filename": version.filename,
                    "external": {
                        "uri": reference.uri,
                        "fingerprint_sha256": reference.fingerprint_sha256,
                        "observed_size_bytes": reference.observed_size_bytes,
                    },
                    "parent_version_id": version.parent_version_id,
                    "parameters": version.parameters,
                    "author_principal_id": version.author_principal_id,
                    "created_at": version.created_at.isoformat(),
                    "activated": activate,
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=command.context.performer_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(stream, expected_revision=stream_revision, events=(event,))
            uow.projections.save_artifact_version(version, activate=activate)
            uow.commit()
            return version


class CreateNoteHandler(_AuthorizedHandler):
    def __call__(self, command: CreateNoteCommand) -> NoteRevision:
        body = require_non_empty(command.body, field="note.body", maximum=100_000)
        with self._uow_factory() as uow:
            project = self._project(uow, command, Permission.ADD_NOTE)
            prior = self._prior(uow, command, "NoteCreated")
            if prior is not None:
                existing = uow.projections.get_note(command.note_id)
                if existing is not None:
                    return existing
            if command.layer_id is not None:
                layer = uow.projections.get_layer(command.layer_id)
                if layer is None or layer.project_id != project.id:
                    raise NotFoundError("Layer was not found in the project")
            now = self._clock.now()
            note = NoteRevision(
                note_id=command.note_id,
                revision=1,
                project_id=project.id,
                author_principal_id=command.context.actor.id,
                body=body,
                recorded_at=now,
                layer_id=command.layer_id,
                frame_id=command.frame_id,
            )
            stream = _note_stream(note.note_id)
            event = EventEnvelope.create(
                stream_id=stream,
                project_id=project.id,
                revision=1,
                event_type="NoteCreated",
                payload={
                    "note_id": note.note_id,
                    "revision": note.revision,
                    "body": note.body,
                    "layer_id": note.layer_id,
                    "frame_id": note.frame_id,
                    "author_principal_id": note.author_principal_id,
                    "recorded_at": note.recorded_at.isoformat(),
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=command.context.performer_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(stream, expected_revision=0, events=(event,))
            uow.projections.save_note(note)
            uow.commit()
            return note


class ReviseNoteHandler(_AuthorizedHandler):
    def __call__(self, command: ReviseNoteCommand) -> NoteRevision:
        body = require_non_empty(command.body, field="note.body", maximum=100_000)
        with self._uow_factory() as uow:
            self._project(uow, command, Permission.ADD_NOTE)
            prior = self._prior(uow, command, "NoteRevised")
            current = uow.projections.get_note(command.note_id)
            if current is None or current.project_id != command.project_id:
                raise NotFoundError("Note was not found")
            if prior is not None:
                return current
            if current.revision != command.expected_revision:
                raise ConcurrencyError(
                    f"Expected note revision {command.expected_revision}, found {current.revision}"
                )
            now = self._clock.now()
            revised = NoteRevision(
                note_id=current.note_id,
                revision=current.revision + 1,
                project_id=current.project_id,
                author_principal_id=command.context.actor.id,
                body=body,
                recorded_at=now,
                layer_id=current.layer_id,
                frame_id=current.frame_id,
            )
            stream = _note_stream(current.note_id)
            event = EventEnvelope.create(
                stream_id=stream,
                project_id=current.project_id,
                revision=revised.revision,
                event_type="NoteRevised",
                payload={
                    "note_id": revised.note_id,
                    "revision": revised.revision,
                    "body": revised.body,
                    "layer_id": revised.layer_id,
                    "frame_id": revised.frame_id,
                    "author_principal_id": revised.author_principal_id,
                    "recorded_at": revised.recorded_at.isoformat(),
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=command.context.performer_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(
                stream,
                expected_revision=current.revision,
                events=(event,),
            )
            uow.projections.save_note(revised)
            uow.commit()
            return revised


__all__ = [
    "ActivateArtifactVersionHandler",
    "AddExternalArtifactVersionHandler",
    "ArchiveArtifactSeriesHandler",
    "CreateNoteHandler",
    "RenameArtifactSeriesHandler",
    "ReviseNoteHandler",
]
