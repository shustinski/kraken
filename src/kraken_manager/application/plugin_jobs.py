"""Durable plugin-job submission and safe immutable result import."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    CancelPluginJobCommand,
    ImportPluginResultCommand,
    PluginResultImport,
    RetryPluginJobCommand,
    SubmitPluginJobCommand,
    SynchronizePluginJobCommand,
)
from kraken_manager.application.errors import AuthorizationError, ConflictError, NotFoundError
from kraken_manager.application.ports import (
    Clock,
    PluginJobGateway,
    PluginResultContentReader,
    StorageProfileCatalog,
    UnitOfWork,
    UnitOfWorkFactory,
)
from kraken_manager.domain.artifacts import (
    ArtifactSeries,
    ArtifactVersion,
    deterministic_frame_series_id,
)
from kraken_manager.domain.common import ArtifactVersionId, FrameId, PluginJobId
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import ProjectState, RepresentationKind, StructureState
from kraken_manager.domain.selection import FrameSelectionV1
from kraken_manager.domain.workflows import (
    PluginFrameOutcome,
    PluginInputV1,
    PluginJob,
    PluginJobManifestV1,
    PluginJobState,
    PluginResultManifestV1,
    PluginResultOutcome,
)


def _job_stream(job_id: PluginJobId | str) -> str:
    return f"plugin-job:{job_id}"


def _series_stream(series_id: object) -> str:
    return f"artifact-series:{series_id}"


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    return value


def _manifest_payload(manifest: PluginJobManifestV1) -> dict[str, object]:
    return {
        "schema_version": manifest.SCHEMA_VERSION,
        "protocol_version": manifest.protocol_version,
        "job_id": str(manifest.job_id),
        "project_id": str(manifest.project_id),
        "layer_id": str(manifest.layer_id),
        "target_representation_id": str(manifest.target_representation_id),
        "selection": manifest.selection.to_dict(),
        "actor_principal_id": str(manifest.actor_principal_id),
        "capability": manifest.capability,
        "inputs": [
            {
                "frame_id": str(item.frame_id),
                "artifact_version_id": str(item.artifact_version_id),
                "sha256": item.sha256,
                "relative_path": item.relative_path,
            }
            for item in manifest.inputs
        ],
        "parameters": _plain(manifest.parameters),
    }


def _manifest_from_payload(payload: Mapping[str, object]) -> PluginJobManifestV1:
    raw_inputs = payload.get("inputs")
    raw_selection = payload.get("selection")
    if not isinstance(raw_inputs, (tuple, list)) or not isinstance(raw_selection, Mapping):
        raise ConflictError("Stored plugin job manifest is damaged")
    try:
        inputs = tuple(
            PluginInputV1(
                frame_id=FrameId(str(item["frame_id"])),
                artifact_version_id=ArtifactVersionId(str(item["artifact_version_id"])),
                sha256=str(item["sha256"]),
                relative_path=str(item["relative_path"]),
            )
            for item in raw_inputs
            if isinstance(item, Mapping)
        )
        if len(inputs) != len(raw_inputs):
            raise ValueError("invalid input row")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("invalid parameters")
        return PluginJobManifestV1(
            job_id=PluginJobId(str(payload["job_id"])),
            project_id=str(payload["project_id"]),  # type: ignore[arg-type]
            layer_id=str(payload["layer_id"]),  # type: ignore[arg-type]
            target_representation_id=str(payload["target_representation_id"]),  # type: ignore[arg-type]
            selection=FrameSelectionV1.from_dict(raw_selection),
            actor_principal_id=str(payload["actor_principal_id"]),  # type: ignore[arg-type]
            capability=str(payload["capability"]),
            inputs=inputs,
            parameters=dict(parameters),
            protocol_version=str(payload.get("protocol_version", "1.0")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConflictError("Stored plugin job manifest is damaged") from exc


def _job_payload(job: PluginJob) -> dict[str, object]:
    return {
        "id": str(job.id),
        "project_id": str(job.project_id),
        "layer_id": str(job.layer_id),
        "selection": job.selection.to_dict(),
        "actor_principal_id": str(job.actor_principal_id),
        "target_representation_id": str(job.target_representation_id),
        "capability": job.capability,
        "state": job.state.value,
        "revision": job.revision,
        "progress": job.progress,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "finished_at": None if job.finished_at is None else job.finished_at.isoformat(),
        "error": job.error,
    }


def _canonical_fingerprint(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(_plain(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _submission_fingerprint(payload: Mapping[str, object]) -> str:
    """Fingerprint a submit command independently from its generated job ID."""

    semantic = dict(payload)
    semantic.pop("job_id", None)
    return _canonical_fingerprint(semantic)


def _result_payload(manifest: PluginResultManifestV1) -> dict[str, object]:
    return {
        "schema_version": manifest.SCHEMA_VERSION,
        "protocol_version": manifest.protocol_version,
        "job_id": str(manifest.job_id),
        "plugin_name": manifest.plugin_name,
        "plugin_version": manifest.plugin_version,
        "outcome": manifest.outcome.value,
        "parameters_applied": _plain(manifest.parameters_applied),
        "results": [
            {
                "output_id": item.output_id,
                "frame_id": str(item.frame_id),
                "outcome": item.outcome.value,
                "relative_path": item.relative_path,
                "sha256": item.sha256,
                "media_type": item.media_type,
                "role": item.role,
                "warning": item.warning,
                "error": item.error,
            }
            for item in manifest.results
        ],
    }


def _phase_key(base: str, phase: str) -> str:
    candidate = f"{base}:{phase}"
    if len(candidate) <= 255:
        return candidate
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    return f"{base[:180]}:{digest}"


def _move_job(job: PluginJob, target: PluginJobState, *, at: object, progress: float | None = None) -> PluginJob:
    """Move through the legal state graph when an Agent callback catches us up."""

    if job.state is target:
        return job
    if job.state in {PluginJobState.SUCCEEDED, PluginJobState.FAILED, PluginJobState.CANCELLED}:
        raise ConflictError(f"Plugin job is already terminal ({job.state.value})")
    timestamp = at  # kept untyped here so Clock implementations remain minimal
    if target is not PluginJobState.IMPORTING and job.state is not PluginJobState.IMPORTING:
        job = _move_job(job, PluginJobState.IMPORTING, at=timestamp)
    elif target is PluginJobState.IMPORTING:
        if job.state is PluginJobState.QUEUED:
            job = job.transition(PluginJobState.STAGING, at=timestamp)
        if job.state is PluginJobState.STAGING:
            job = job.transition(PluginJobState.RUNNING, at=timestamp)
        if job.state is PluginJobState.WAITING_FOR_USER:
            job = job.transition(PluginJobState.RUNNING, at=timestamp)
        if job.state in {PluginJobState.RUNNING, PluginJobState.PARTIAL, PluginJobState.AWAITING_AUTHORIZATION}:
            job = job.transition(PluginJobState.IMPORTING, at=timestamp)
        if job.state is not PluginJobState.IMPORTING:
            raise ConflictError(f"Plugin job cannot start importing from {job.state.value}")
        return job
    return job.transition(target, at=timestamp, progress=progress)


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
            }.get(command.capability)
            if required_kind is not None and target.kind is not required_kind:
                raise ConflictError(f"Plugin capability requires a {required_kind.value} target representation")
            command.selection.validate_bounds(width=project.width, height=project.height)
            manifest = PluginJobManifestV1(
                job_id=command.job_id,
                project_id=project.id,
                layer_id=layer.id,
                target_representation_id=target.id,
                selection=command.selection,
                actor_principal_id=command.context.actor.id,
                capability=command.capability,
                inputs=command.inputs,
                parameters=command.parameters,
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
                    or series.frame_id != item.frame_id
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


class ImportPluginResultHandler:
    """Validate an Agent result and commit immutable artifact versions atomically."""

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

    def __call__(self, command: ImportPluginResultCommand) -> PluginResultImport:
        manifest = command.manifest
        fingerprint = _canonical_fingerprint(_result_payload(manifest))
        with self._uow_factory() as uow:
            job = uow.projections.get_plugin_job(manifest.job_id)
            if job is None:
                raise NotFoundError("Plugin job was not found")
            if command.context.actor.id != job.actor_principal_id:
                raise AuthorizationError("Only the initiating principal may authorize plugin result import")
            original = self._created_manifest(uow, job)
            if manifest.job_id != original.job_id or manifest.protocol_version != original.protocol_version:
                raise ConflictError("Plugin result does not match the submitted job/protocol")

            imported = self._event_for_fingerprint(uow, job, "PluginResultImported", fingerprint)
            if imported is not None:
                versions: list[ArtifactVersion] = []
                for identifier in imported.payload.get("artifact_version_ids", ()):
                    version = uow.projections.get_artifact_version(ArtifactVersionId(str(identifier)))
                    if version is None:
                        raise ConflictError("Imported artifact projection is missing and must be rebuilt")
                    versions.append(version)
                return PluginResultImport(job, tuple(versions), already_imported=True)
            terminal_event = {
                PluginResultOutcome.FAILED: "PluginJobFailed",
                PluginResultOutcome.CANCELLED: "PluginJobCancelled",
            }.get(manifest.outcome)
            if terminal_event is not None:
                if self._event_for_fingerprint(uow, job, terminal_event, fingerprint) is not None:
                    return PluginResultImport(job, already_imported=True)
                if job.state in {
                    PluginJobState.SUCCEEDED,
                    PluginJobState.FAILED,
                    PluginJobState.CANCELLED,
                }:
                    raise ConflictError("A different terminal result was already recorded for this plugin job")
                target_state = (
                    PluginJobState.FAILED
                    if manifest.outcome is PluginResultOutcome.FAILED
                    else PluginJobState.CANCELLED
                )
                job = _move_job(job, target_state, at=self._clock.now())
                self._save_job_event(
                    uow,
                    job=job,
                    command=command,
                    event_type=terminal_event,
                    fingerprint=fingerprint,
                    phase=manifest.outcome.value,
                    extra={"result": _result_payload(manifest)},
                )
                uow.commit()
                return PluginResultImport(job)
            if job.state in {
                PluginJobState.SUCCEEDED,
                PluginJobState.FAILED,
                PluginJobState.CANCELLED,
            }:
                raise ConflictError("A different result was already committed for this plugin job")

            inputs_by_frame = {item.frame_id: item for item in original.inputs}
            for result in manifest.results:
                if result.frame_id not in inputs_by_frame:
                    raise ConflictError(f"Plugin returned unknown frame {result.frame_id}")
                required_output = {
                    "frames.vectorize.v1": ("application/x-cif", "vector"),
                    "frames.binary-segment.v1": ("image/png", "binary-image"),
                }.get(original.capability)
                if (
                    result.outcome is PluginFrameOutcome.SUCCEEDED
                    and required_output is not None
                    and (result.media_type, result.role) != required_output
                ):
                    raise ConflictError("Plugin output format/role does not match the requested capability")
            result_frames = {item.frame_id for item in manifest.results}
            is_partial = manifest.outcome is PluginResultOutcome.PARTIAL or result_frames != set(inputs_by_frame) or any(
                item.outcome is not PluginFrameOutcome.SUCCEEDED for item in manifest.results
            )

            project = uow.projections.get_project(job.project_id)
            if project is None:
                raise NotFoundError("Plugin job project was not found")
            if project.state is ProjectState.ARCHIVED:
                raise ConflictError("Archived project is read-only")
            profile = self._storage_profiles.get(project.storage_profile)
            if profile is None:
                raise NotFoundError("Project storage profile was not found")
            decision = self._authorization.decide(
                principal=command.context.actor,
                storage=profile,
                permission=Permission.RUN_PLUGIN,
                roles=uow.acl.roles_for(project.id, command.context.actor.id),
                gitlab_identity_verified=command.context.gitlab_identity_verified,
            )
            if not decision.allowed and decision.code == "gitlab_live_check_required":
                if self._event_for_fingerprint(uow, job, "PluginResultAwaitingAuthorization", fingerprint) is None:
                    job = _move_job(job, PluginJobState.AWAITING_AUTHORIZATION, at=self._clock.now())
                    self._save_job_event(
                        uow,
                        job=job,
                        command=command,
                        event_type="PluginResultAwaitingAuthorization",
                        fingerprint=fingerprint,
                        phase="awaiting-authorization",
                    )
                    uow.commit()
                return PluginResultImport(job, awaiting_authorization=True)
            decision.require()

            if is_partial and not command.confirm_partial:
                if self._event_for_fingerprint(uow, job, "PluginPartialResultReceived", fingerprint) is None:
                    job = _move_job(job, PluginJobState.PARTIAL, at=self._clock.now())
                    self._save_job_event(
                        uow,
                        job=job,
                        command=command,
                        event_type="PluginPartialResultReceived",
                        fingerprint=fingerprint,
                        phase="partial-preview",
                    )
                    uow.commit()
                return PluginResultImport(job, requires_partial_confirmation=True)

            target = uow.projections.get_representation(job.target_representation_id)
            if (
                target is None
                or target.project_id != project.id
                or target.layer_id != job.layer_id
                or target.state is StructureState.ARCHIVED
            ):
                raise ConflictError("Target representation is no longer writable")
            now = self._clock.now()
            job = _move_job(job, PluginJobState.IMPORTING, at=now)
            versions: list[ArtifactVersion] = []
            wanted_frames = {result.frame_id for result in manifest.results}
            coordinates_by_frame = {}
            for coordinate in job.selection.iter_coordinates():
                frame_id = coordinate.frame_id(project.id)
                if frame_id in wanted_frames:
                    coordinates_by_frame[frame_id] = (coordinate.x, coordinate.y)
                    if len(coordinates_by_frame) == len(wanted_frames):
                        break
            for result in manifest.results:
                if result.outcome is not PluginFrameOutcome.SUCCEEDED:
                    continue
                assert result.relative_path is not None
                assert result.sha256 is not None
                assert result.media_type is not None
                input_item = inputs_by_frame[result.frame_id]
                input_version = uow.projections.get_artifact_version(input_item.artifact_version_id)
                if input_version is None or input_version.sha256 != input_item.sha256:
                    raise ConflictError("A snapshotted plugin input version is missing or damaged")
                series_id = deterministic_frame_series_id(target.id, result.frame_id)
                series = uow.projections.get_artifact_series(series_id)
                # Domain validation guarantees a normalized relative POSIX
                # manifest path; application code must not depend on pathlib.
                filename = result.relative_path.rsplit("/", 1)[-1]
                new_series = series is None
                if new_series:
                    series = ArtifactSeries.for_frame(
                        series_id=series_id,
                        project_id=project.id,
                        layer_id=job.layer_id,
                        representation_id=target.id,
                        frame_id=result.frame_id,
                        name=filename,
                    )
                elif (
                    series.archived
                    or series.project_id != project.id
                    or series.layer_id != job.layer_id
                    or series.representation_id != target.id
                    or series.frame_id != result.frame_id
                ):
                    raise ConflictError("Canonical output artifact series has conflicting metadata")

                existing = uow.projections.get_artifact_version(ArtifactVersionId(result.output_id))
                if existing is not None:
                    # A legitimate repeated callback was returned above from
                    # PluginResultImported. Without that receipt, accepting an
                    # arbitrary existing UUID would let a plugin claim a
                    # manually-created artifact as its own output.
                    raise ConflictError("Plugin output ID already belongs to an unassociated artifact version")
                try:
                    stored = uow.blobs.put(
                        self._content_reader.iter_output(manifest, result.relative_path),
                        expected_sha256=result.sha256,
                    )
                except (FileNotFoundError, OSError, ValueError) as exc:
                    raise ConflictError(f"Plugin output {result.relative_path!r} is missing or has a bad SHA-256") from exc
                active = uow.projections.get_active_artifact_version(series.id)
                parameters = dict(manifest.parameters_applied)
                parameters["plugin_output_role"] = result.role
                coordinate = coordinates_by_frame.get(result.frame_id)
                if coordinate is not None:
                    parameters["x"], parameters["y"] = coordinate
                version = ArtifactVersion.managed(
                    version_id=ArtifactVersionId(result.output_id),
                    series_id=series.id,
                    blob=stored.blob,
                    media_type=result.media_type,
                    filename=filename,
                    author_principal_id=command.context.actor.id,
                    created_at=now,
                    parent_version_id=None if active is None else active.id,
                    input_version_ids=(input_item.artifact_version_id,),
                    tool_name=manifest.plugin_name,
                    tool_version=manifest.plugin_version,
                    parameters=parameters,
                )
                stream = _series_stream(series.id)
                revision = uow.event_store.current_revision(stream)
                events: list[EventEnvelope] = []
                if new_series:
                    events.append(
                        EventEnvelope.create(
                            stream_id=stream,
                            project_id=project.id,
                            revision=revision + 1,
                            event_type="ArtifactSeriesCreated",
                            payload={
                                "artifact_series_id": str(series.id),
                                "scope": series.scope.value,
                                "name": series.name,
                                "layer_id": str(series.layer_id),
                                "representation_id": str(series.representation_id),
                                "frame_id": str(series.frame_id),
                                "archived": False,
                            },
                            actor=ActorSnapshot.from_principal(command.context.actor),
                            recorded_at=now,
                            performer_id=command.context.performer_id,
                            program=ProgramSnapshot(manifest.plugin_name, manifest.plugin_version),
                            correlation_id=command.context.correlation_id,
                        )
                    )
                events.append(
                    EventEnvelope.create(
                        stream_id=stream,
                        project_id=project.id,
                        revision=revision + len(events) + 1,
                        event_type="ArtifactVersionCreated",
                        payload={
                            "artifact_version_id": str(version.id),
                            "series_id": str(series.id),
                            "layer_id": str(series.layer_id),
                            "representation_id": str(series.representation_id),
                            "frame_id": str(series.frame_id),
                            "sha256": version.sha256,
                            "size_bytes": version.size_bytes,
                            "media_type": version.media_type,
                            "filename": version.filename,
                            "blob": {"sha256": version.blob.sha256, "size_bytes": version.blob.size_bytes},
                            "parent_version_id": None if version.parent_version_id is None else str(version.parent_version_id),
                            "input_version_ids": [str(item) for item in version.input_version_ids],
                            "tool_name": version.tool_name,
                            "tool_version": version.tool_version,
                            "parameters": _plain(version.parameters),
                            "author_principal_id": str(version.author_principal_id),
                            "created_at": version.created_at.isoformat(),
                            "activated": True,
                            "branched": False,
                            "plugin_job_id": str(job.id),
                        },
                        actor=ActorSnapshot.from_principal(command.context.actor),
                        recorded_at=now,
                        performer_id=command.context.performer_id,
                        program=ProgramSnapshot(manifest.plugin_name, manifest.plugin_version),
                        correlation_id=command.context.correlation_id,
                    )
                )
                uow.event_store.append(stream, expected_revision=revision, events=tuple(events))
                if new_series:
                    uow.projections.save_artifact_series(series)
                uow.projections.save_artifact_version(version, activate=True)
                versions.append(version)

            progress = len(versions) / len(original.inputs)
            final_state = PluginJobState.PARTIAL if is_partial else PluginJobState.SUCCEEDED
            job = _move_job(job, final_state, at=now, progress=progress)
            self._save_job_event(
                uow,
                job=job,
                command=command,
                event_type="PluginResultImported",
                fingerprint=fingerprint,
                phase="import",
                extra={
                    "partial": is_partial,
                    "artifact_version_ids": [str(version.id) for version in versions],
                    "result": _result_payload(manifest),
                },
            )
            uow.commit()
            return PluginResultImport(job, tuple(versions))


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
            if job.state is target:
                return job
            now = self._clock.now()
            changed = job
            if target in {PluginJobState.STAGING, PluginJobState.RUNNING}:
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


__all__ = [
    "CancelPluginJobHandler",
    "ImportPluginResultHandler",
    "RetryPluginJobHandler",
    "SubmitPluginJobHandler",
    "SynchronizePluginJobHandler",
]
