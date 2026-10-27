"""plugin_jobs: submit plugin job handler."""
from __future__ import annotations
from collections.abc import Mapping
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    SubmitPluginJobCommand,
)
from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.application.ports import (
    Clock,
    PluginJobGateway,
    StorageProfileCatalog,
    UnitOfWorkFactory,
)
from kraken_manager.domain.artifacts import (
    ArtifactScope,
    is_training_dataset_artifact,
)
from kraken_manager.domain.common import PluginJobId
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import ProjectState, RepresentationKind, StructureState
from kraken_manager.domain.workflows import (
    PluginJob,
    PluginJobManifestV1,
)

from kraken_manager.application.plugin_jobs_parts.manifest_payload import (
    _manifest_from_payload,
    _manifest_payload,
)

from kraken_manager.application.plugin_jobs_parts.plain import _job_stream

from kraken_manager.application.plugin_jobs_parts.result_payload import (
    _canonical_fingerprint,
    _job_payload,
    _submission_fingerprint,
)

class SubmitPluginJobHandler:
    """Persist the authoritative job first, then idempotently enqueue it in Agent."""

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

    def __call__(self, command: SubmitPluginJobCommand) -> PluginJob:
        if not self._gateway.is_available(command.capability, command.protocol_version):
            raise ConflictError("No compatible Kraken Agent/plugin capability is available")
        manifest: PluginJobManifestV1 | None = None
        job: PluginJob | None = None
        with self._uow_factory() as uow:
            project = uow.projections.get_project(command.project_id)
            if project is None:
                raise NotFoundError("Project was not found")
            if project.state is ProjectState.ARCHIVED:
                raise ConflictError("Archived project is read-only")
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
            layer = uow.projections.get_layer(command.layer_id)
            if layer is None or layer.project_id != project.id or layer.state is StructureState.ARCHIVED:
                raise NotFoundError("Active layer was not found in the project")
            target = uow.projections.get_representation(command.target_representation_id)
            if (
                target is None
                or target.project_id != project.id
                or target.layer_id != layer.id
                or target.state is StructureState.ARCHIVED
            ):
                raise NotFoundError("Active target representation was not found in the layer")
            required_kind = {
                "frames.vectorize.v1": RepresentationKind.VECTOR,
                "frames.binary-segment.v1": RepresentationKind.IMAGE,
                "frames.binary-segment.v2": RepresentationKind.IMAGE,
                "frames.confidence.v1": RepresentationKind.IMAGE,
            }.get(command.capability)
            if required_kind is not None and target.kind is not required_kind:
                raise ConflictError(f"Plugin capability requires a {required_kind.value} target representation")
            command.selection.validate_bounds(width=project.width, height=project.height)
            parameters = dict(command.parameters)
            if "model_version_id" in parameters:
                from kraken_manager.application.model_inputs import resolve_model_input
                from kraken_manager.application.quality import QualityService
                if command.capability not in {"frames.binary-segment.v2", "frames.confidence.v1"}:
                    raise ConflictError("This capability has no managed model input port")
                model, _, metadata = resolve_model_input(uow.projections, project.id, layer.id, parameters)
                QualityService.require_clear_inputs(uow.event_store, project.id, (model.id,), projections=uow.projections)
                parameters.update(metadata)
            manifest = PluginJobManifestV1(
                job_id=command.job_id,
                project_id=project.id,
                layer_id=layer.id,
                target_representation_id=target.id,
                selection=command.selection,
                actor_principal_id=command.context.actor.id,
                capability=command.capability,
                inputs=command.inputs,
                parameters=parameters,
                protocol_version=command.protocol_version,
            )
            if manifest.selection.cardinality() != len(manifest.inputs):
                raise ConflictError("Plugin inputs must cover the frame selection exactly")
            remaining = {item.frame_id for item in manifest.inputs}
            for coordinate in manifest.selection.iter_coordinates():
                remaining.discard(coordinate.frame_id(project.id))
                if not remaining:
                    break
            if remaining:
                raise ConflictError("Plugin input contains a frame outside the immutable selection")
            for item in manifest.inputs:
                version = uow.projections.get_artifact_version(item.artifact_version_id)
                if version is None or version.sha256 != item.sha256:
                    raise ConflictError("Plugin input version/hash does not match Kraken state")
                series = uow.projections.get_artifact_series(version.series_id)
                if (
                    series is None
                    or series.archived
                    or series.project_id != project.id
                    or series.layer_id != layer.id
                    or (
                        series.frame_id != item.frame_id
                        and not (
                            command.capability == "dataset.model.train.v1"
                            and series.scope is ArtifactScope.LAYER_ATTACHMENT
                            and is_training_dataset_artifact(version)
                        )
                    )
                ):
                    raise ConflictError("Plugin input artifact does not belong to the declared layer/frame")

            payload = _manifest_payload(manifest)
            fingerprint = _canonical_fingerprint(payload)
            request_fingerprint = _submission_fingerprint(payload)
            prior = [
                event
                for event in uow.event_store.find_by_idempotency_key(project.id, command.context.idempotency_key)
                if event.event_type == "PluginJobCreated"
            ]
            if prior:
                stored_payload = prior[-1].payload.get("manifest")
                if not isinstance(stored_payload, Mapping):
                    raise ConflictError("Stored plugin job manifest is damaged")
                if _submission_fingerprint(stored_payload) != request_fingerprint:
                    raise ConflictError("Idempotency key was reused with a different plugin manifest")
                manifest = _manifest_from_payload(stored_payload)
                job = uow.projections.get_plugin_job(PluginJobId(str(prior[-1].payload["plugin_job_id"])))
                if job is None:
                    raise ConflictError("Plugin job projection is missing and must be rebuilt")
            else:
                if uow.projections.get_plugin_job(command.job_id) is not None:
                    raise ConflictError("Plugin job ID already exists")
                now = self._clock.now()
                job = PluginJob.create(
                    job_id=command.job_id,
                    project_id=project.id,
                    layer_id=layer.id,
                    selection=manifest.selection,
                    actor_principal_id=manifest.actor_principal_id,
                    target_representation_id=target.id,
                    capability=manifest.capability,
                    created_at=now,
                )
                event = EventEnvelope.create(
                    stream_id=_job_stream(job.id),
                    project_id=project.id,
                    revision=1,
                    event_type="PluginJobCreated",
                    payload={
                        "plugin_job_id": str(job.id),
                        "manifest": payload,
                        "manifest_fingerprint": fingerprint,
                        "request_fingerprint": request_fingerprint,
                        "job": _job_payload(job),
                    },
                    actor=ActorSnapshot.from_principal(command.context.actor),
                    recorded_at=now,
                    effective_at=command.context.effective_at,
                    performer_id=command.context.performer_id,
                    correlation_id=command.context.correlation_id,
                    idempotency_key=command.context.idempotency_key,
                )
                uow.event_store.append(_job_stream(job.id), expected_revision=0, events=(event,))
                uow.projections.save_plugin_job(job)
                uow.commit()
        assert manifest is not None and job is not None
        # Agent enqueue is itself idempotent. If the process dies between the
        # commit and this call, replaying the same command repairs the handoff.
        self._gateway.submit(manifest)
        return job
