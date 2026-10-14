"""plugin_jobs: retry plugin job handler."""
from __future__ import annotations
from collections.abc import Mapping
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    RetryPluginJobCommand,
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

from kraken_manager.application.plugin_jobs_parts.manifest_payload import _manifest_from_payload

from kraken_manager.application.plugin_jobs_parts.plain import _job_stream

from kraken_manager.application.plugin_jobs_parts.result_payload import _job_payload

class RetryPluginJobHandler:
    """Re-enqueue the immutable original manifest after Agent state loss."""

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

    def __call__(self, command: RetryPluginJobCommand) -> PluginJob:
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
            if job.state is not PluginJobState.RECOVERY_REQUIRED:
                raise ConflictError("Only a recovery-required plugin job can be retried")
            created = [
                event
                for event in uow.event_store.load_stream(_job_stream(job.id))
                if event.event_type == "PluginJobCreated"
            ]
            if len(created) != 1 or not isinstance(
                created[0].payload.get("manifest"),
                Mapping,
            ):
                raise ConflictError("Authoritative plugin job manifest is missing")
            manifest = _manifest_from_payload(created[0].payload["manifest"])  # type: ignore[arg-type]

        if not self._gateway.is_available(
            manifest.capability,
            manifest.protocol_version,
        ):
            raise ConflictError("No compatible Kraken Agent/plugin capability is available")
        self._gateway.submit(manifest)

        with self._uow_factory() as uow:
            current = uow.projections.get_plugin_job(command.job_id)
            if current is None or current.revision != command.expected_revision:
                raise ConflictError("Plugin job revision changed while retrying")
            now = self._clock.now()
            retried = current.transition(
                PluginJobState.QUEUED,
                at=now,
                progress=0,
            )
            stream = _job_stream(retried.id)
            stream_revision = uow.event_store.current_revision(stream)
            event = EventEnvelope.create(
                stream_id=stream,
                project_id=retried.project_id,
                revision=stream_revision + 1,
                event_type="PluginJobRetried",
                payload={
                    "plugin_job_id": str(retried.id),
                    "job": _job_payload(retried),
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
            uow.projections.save_plugin_job(retried)
            uow.commit()
            return retried
