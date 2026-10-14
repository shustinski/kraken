"""plugin_jobs: save job event."""
from __future__ import annotations
from collections.abc import Mapping
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    ImportPluginResultCommand,
)
from kraken_manager.application.errors import ConflictError
from kraken_manager.application.ports import (
    Clock,
    PluginResultContentReader,
    StorageProfileCatalog,
    UnitOfWork,
    UnitOfWorkFactory,
)
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
from kraken_manager.domain.workflows import (
    PluginJob,
    PluginJobManifestV1,
)

from kraken_manager.application.plugin_jobs_parts.manifest_payload import _manifest_from_payload

from kraken_manager.application.plugin_jobs_parts.plain import _job_stream

from kraken_manager.application.plugin_jobs_parts.result_payload import (
    _job_payload,
    _phase_key,
)


class SaveJobEventMixin:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        storage_profiles: StorageProfileCatalog,
        clock: Clock,
        content_reader: PluginResultContentReader,
        authorization: AuthorizationPolicy | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._storage_profiles = storage_profiles
        self._clock = clock
        self._content_reader = content_reader
        self._authorization = authorization or AuthorizationPolicy()

    @staticmethod
    def _created_manifest(uow: UnitOfWork, job: PluginJob) -> PluginJobManifestV1:
        created = [
            event
            for event in uow.event_store.load_stream(_job_stream(job.id))
            if event.event_type == "PluginJobCreated"
        ]
        if len(created) != 1 or not isinstance(created[0].payload.get("manifest"), Mapping):
            raise ConflictError("Authoritative plugin job manifest is missing")
        return _manifest_from_payload(created[0].payload["manifest"])  # type: ignore[arg-type]

    @staticmethod
    def _event_for_fingerprint(uow: UnitOfWork, job: PluginJob, event_type: str, fingerprint: str):
        return next(
            (
                event
                for event in reversed(uow.event_store.load_stream(_job_stream(job.id)))
                if event.event_type == event_type
                and str(event.payload.get("result_fingerprint")) == fingerprint
            ),
            None,
        )

    def _save_job_event(
        self,
        uow: UnitOfWork,
        *,
        job: PluginJob,
        command: ImportPluginResultCommand,
        event_type: str,
        fingerprint: str,
        phase: str,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        stream = _job_stream(job.id)
        revision = uow.event_store.current_revision(stream)
        payload = {
            "plugin_job_id": str(job.id),
            "result_fingerprint": fingerprint,
            "job": _job_payload(job),
            **dict(extra or {}),
        }
        event = EventEnvelope.create(
            stream_id=stream,
            project_id=job.project_id,
            revision=revision + 1,
            event_type=event_type,
            payload=payload,
            actor=ActorSnapshot.from_principal(command.context.actor),
            recorded_at=self._clock.now(),
            effective_at=command.context.effective_at,
            performer_id=command.context.performer_id,
            program=ProgramSnapshot(command.manifest.plugin_name, command.manifest.plugin_version),
            correlation_id=command.context.correlation_id,
            idempotency_key=_phase_key(command.context.idempotency_key, phase),
        )
        uow.event_store.append(stream, expected_revision=revision, events=(event,))
        uow.projections.save_plugin_job(job)
