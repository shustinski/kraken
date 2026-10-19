"""persistent_services: append auxiliary event."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.errors import (
    ConcurrencyError as ApplicationConcurrencyError,
)
from kraken_manager.application.errors import (
    NotFoundError as ApplicationNotFoundError,
)
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
from kraken_manager.domain.identity import Permission
from kraken_manager.infrastructure.filesystem._codec import encode_model
from kraken_server.services import CommandContext

from kraken_server.persistent_services_parts.layer_source_scan import _project_id


class AppendAuxiliaryEventMixin:
    def _append_auxiliary_event(
        self,
        project_id: str,
        *,
        stream_id: str,
        event_type: str,
        payload: Mapping[str, object],
        program: ProgramSnapshot,
        context: CommandContext,
    ) -> dict[str, Any]:
        actor = self._actor(context.actor_id)
        identifier = _project_id(project_id)
        try:
            with self.uow_factory() as uow:
                project = uow.projections.get_project(identifier)
                if project is None:
                    raise ApplicationNotFoundError("Project was not found")
                profile = self.profiles.get(project.storage_profile)
                if profile is None:
                    raise ApplicationNotFoundError("Storage profile was not found")
                AuthorizationPolicy().require(
                    principal=actor,
                    storage=profile,
                    permission=Permission.MANAGE_STRUCTURE if event_type == "LayerStitchingStatusChangedV1" else Permission.RUN_PLUGIN,
                    roles=uow.acl.roles_for(project.id, actor.id),
                    gitlab_identity_verified=True,
                )
                prior = uow.event_store.find_by_idempotency_key(
                    project.id, context.idempotency_key
                )
                if prior:
                    return encode_model(prior[-1])
                if event_type == "LayerStitchingStatusChangedV1":
                    from kraken_manager.domain.common import ArtifactVersionId
                    from kraken_manager.domain.stitching import StitchingKind, validate_stitching_file
                    from kraken_server.services import ValidationError

                    try:
                        kind = StitchingKind(str(payload.get("kind", "")))
                        if not isinstance(payload.get("completed"), bool):
                            raise ValueError("completed must be a boolean")
                        if payload["completed"]:
                            version = uow.projections.get_artifact_version(ArtifactVersionId(str(payload.get("version_id", ""))))
                            series = None if version is None else uow.projections.get_artifact_series(version.series_id)
                            validate_stitching_file(project_id, payload["layer_id"], kind, version, series)
                    except ValueError as exc:
                        raise ValidationError(str(exc)) from exc
                revision = uow.event_store.current_revision(stream_id)
                if context.expected_revision != revision:
                    raise ApplicationConcurrencyError(
                        f"Expected stream revision {context.expected_revision}, found {revision}"
                    )
                event = EventEnvelope.create(
                    stream_id=stream_id,
                    project_id=project.id,
                    revision=revision + 1,
                    event_type=event_type,
                    payload=payload,
                    actor=ActorSnapshot.from_principal(actor),
                    program=program,
                    idempotency_key=context.idempotency_key,
                )
                uow.event_store.append(
                    stream_id, expected_revision=revision, events=(event,)
                )
                uow.commit()
                return encode_model(event)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")
