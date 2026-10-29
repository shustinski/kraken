"""plugin_jobs: synchronize plugin job handler."""
from __future__ import annotations
from dataclasses import replace
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    SynchronizePluginJobCommand,
)
from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.application.ports import (
    Clock,
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

from kraken_manager.application.plugin_jobs_parts.result_payload import (
    _job_payload,
    _move_job,
)

class SynchronizePluginJobHandler:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        storage_profiles: StorageProfileCatalog,
        clock: Clock,
        authorization: AuthorizationPolicy | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._storage_profiles = storage_profiles
        self._clock = clock
        self._authorization = authorization or AuthorizationPolicy()

    def __call__(self, command: SynchronizePluginJobCommand) -> PluginJob:
        target = PluginJobState(command.state)
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
            if job.revision != command.expected_revision:
                raise ConflictError("Plugin job revision changed")
            if job.state is target and (command.progress is None or command.progress == job.progress):
                return job
            now = self._clock.now()
            changed = job
            if job.state is target:
                changed = replace(job, revision=job.revision + 1, updated_at=now)
            elif target in {PluginJobState.STAGING, PluginJobState.RUNNING}:
                if changed.state is PluginJobState.QUEUED:
                    changed = changed.transition(PluginJobState.STAGING, at=now)
                if target is PluginJobState.RUNNING and changed.state is PluginJobState.STAGING:
                    changed = changed.transition(PluginJobState.RUNNING, at=now)
            elif target is PluginJobState.WAITING_FOR_USER:
                if changed.state is PluginJobState.QUEUED:
                    changed = changed.transition(PluginJobState.STAGING, at=now)
                if changed.state is PluginJobState.STAGING:
                    changed = changed.transition(PluginJobState.RUNNING, at=now)
                if changed.state is PluginJobState.RUNNING:
                    changed = changed.transition(PluginJobState.WAITING_FOR_USER, at=now)
                if changed.state is not PluginJobState.WAITING_FOR_USER:
                    raise ConflictError(
                        f"Plugin job cannot wait for user from {changed.state.value}"
                    )
            else:
                changed = _move_job(
                    changed,
                    target,
                    at=now,
                    progress=command.progress,
                )
            if command.progress is not None:
                changed = replace(changed, progress=max(changed.progress, command.progress))
            if command.error and target in {
                PluginJobState.FAILED,
                PluginJobState.RECOVERY_REQUIRED,
            }:
                changed = replace(changed, error=command.error[:10_000])
            stream = _job_stream(changed.id)
            stream_revision = uow.event_store.current_revision(stream)
            event = EventEnvelope.create(
                stream_id=stream,
                project_id=changed.project_id,
                revision=stream_revision + 1,
                event_type="PluginJobSynchronized",
                payload={
                    "plugin_job_id": str(changed.id),
                    "job": _job_payload(changed),
                    "agent_state": target.value,
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
            uow.projections.save_plugin_job(changed)
            uow.commit()
            return changed
