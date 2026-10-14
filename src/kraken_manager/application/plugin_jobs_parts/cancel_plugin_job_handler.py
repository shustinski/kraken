"""plugin_jobs: cancel plugin job handler."""
from __future__ import annotations
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    CancelPluginJobCommand,
)
from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.application.ports import (
    Clock,
    PluginJobGateway,
    StorageProfileCatalog,
    UnitOfWorkFactory,
)
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.workflows import (
    PluginJob,
    PluginJobState,
)

from kraken_manager.application.plugin_jobs_parts.plain import _job_stream

from kraken_manager.application.plugin_jobs_parts.result_payload import _job_payload

class CancelPluginJobHandler:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        storage_profiles: StorageProfileCatalog,
        clock: Clock,
        gateway: PluginJobGateway,
        authorization: AuthorizationPolicy | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._storage_profiles = storage_profiles
        self._clock = clock
        self._gateway = gateway
        self._authorization = authorization or AuthorizationPolicy()

    def __call__(self, command: CancelPluginJobCommand) -> PluginJob:
        with self._uow_factory() as uow:
            project = uow.projections.get_project(command.project_id)
            job = uow.projections.get_plugin_job(command.job_id)
            if project is None or job is None or job.project_id != project.id:
                raise NotFoundError("Plugin job was not found")
            profile = self._storage_profiles.get(project.storage_profile)
            if profile is None:
                raise NotFoundError("Project storage profile was not found")
            self._authorization.require(
                principal=command.context.actor,
                storage=profile,
                permission=Permission.RUN_PLUGIN,
                roles=uow.acl.roles_for(project.id, command.context.actor.id),
                gitlab_identity_verified=command.context.gitlab_identity_verified,
            )
            prior = [
                event
                for event in uow.event_store.find_by_idempotency_key(
                    project.id,
                    command.context.idempotency_key,
                )
                if event.event_type == "PluginJobCancelled"
            ]
            if prior:
                return job
            if job.revision != command.expected_revision:
                raise ConflictError("Plugin job revision changed")
            if job.state in {
                PluginJobState.SUCCEEDED,
                PluginJobState.FAILED,
                PluginJobState.CANCELLED,
            }:
                raise ConflictError("Plugin job is already terminal")
        self._gateway.cancel(job.id)
        with self._uow_factory() as uow:
            current = uow.projections.get_plugin_job(command.job_id)
            if current is None or current.revision != command.expected_revision:
                raise ConflictError("Plugin job revision changed while cancelling")
            now = self._clock.now()
            cancelled = current.transition(PluginJobState.CANCELLED, at=now)
            stream = _job_stream(cancelled.id)
            stream_revision = uow.event_store.current_revision(stream)
            event = EventEnvelope.create(
                stream_id=stream,
                project_id=cancelled.project_id,
                revision=stream_revision + 1,
                event_type="PluginJobCancelled",
                payload={
                    "plugin_job_id": str(cancelled.id),
                    "job": _job_payload(cancelled),
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
                expected_revision=stream_revision,
                events=(event,),
            )
            uow.projections.save_plugin_job(cancelled)
            uow.commit()
            return cancelled
