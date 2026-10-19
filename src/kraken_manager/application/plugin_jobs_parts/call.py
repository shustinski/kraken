"""plugin_jobs: call."""
from __future__ import annotations
from uuid import UUID, uuid5
from kraken_manager.application.dto import (
    ImportPluginResultCommand,
    PluginResultImport,
)
from kraken_manager.application.errors import AuthorizationError, ConflictError, NotFoundError
from kraken_manager.domain.artifacts import (
    ArtifactScope,
    ArtifactSeries,
    ArtifactVersion,
    deterministic_frame_series_id,
)
from kraken_manager.domain.common import ArtifactSeriesId, ArtifactVersionId
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import ProjectState, StructureState
from kraken_manager.domain.workflows import (
    PluginFrameOutcome,
    PluginJobState,
    PluginResultOutcome,
)

from kraken_manager.application.plugin_jobs_parts.plain import (
    _plain,
    _series_stream,
)

from kraken_manager.application.plugin_jobs_parts.result_payload import (
    _canonical_fingerprint,
    _move_job,
    _result_payload,
)


class CallMixin:
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
                errors = manifest.parameters_applied.get("transport_errors", ())
                error = (
                    "; ".join(str(item) for item in errors)
                    if errors
                    else str(manifest.parameters_applied["error"])
                    if manifest.parameters_applied.get("error")
                    else None
                )
                job = _move_job(job, target_state, at=self._clock.now(), error=error)
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
                    "frames.binary-segment.v2": ("image/png", "binary-image"),
                    "frames.confidence.v1": ("image/jpeg", "confidence"),
                    "frames.dataset.prepare.v1": ("application/zip", "dataset"),
                    "layer.confidence.analyze.v1": (
                        "application/vnd.kraken.confidence+json",
                        "confidence",
                    ),
                }.get(original.capability)
                accepted = required_output is not None and (result.media_type, result.role) == required_output
                if original.capability == "dataset.model.train.v1" and result.role == "model":
                    accepted = True
                if original.capability == "dataset.model.train.v1" and result.role != "model":
                    raise ConflictError("Training must return a model artifact")
                if (
                    result.outcome is PluginFrameOutcome.SUCCEEDED
                    and required_output is not None
                    and not accepted
                    and original.capability != "dataset.model.train.v1"
                ):
                    raise ConflictError("Plugin output format/role does not match the requested capability")
            result_frames = {item.frame_id for item in manifest.results}
            dataset_output = original.capability == "frames.dataset.prepare.v1"
            model_output = original.capability == "dataset.model.train.v1"
            aggregate_output = dataset_output or model_output
            complete_selection = len(manifest.results) == 1 if aggregate_output else result_frames == set(inputs_by_frame)
            is_partial = (
                manifest.outcome is PluginResultOutcome.PARTIAL
                or not complete_selection
                or any(item.outcome is not PluginFrameOutcome.SUCCEEDED for item in manifest.results)
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
                series_id = (
                    ArtifactSeriesId(
                        str(
                            uuid5(
                                UUID(str(job.id)),
                                "model-artifact-series" if model_output else "dataset-artifact-series",
                            )
                        )
                    )
                    if aggregate_output
                    else deterministic_frame_series_id(target.id, result.frame_id)
                )
                series = uow.projections.get_artifact_series(series_id)
                # Domain validation guarantees a normalized relative POSIX
                # manifest path; application code must not depend on pathlib.
                filename = result.relative_path.rsplit("/", 1)[-1]
                new_series = series is None
                expected_series = (
                    ArtifactSeries(
                        id=series_id,
                        project_id=project.id,
                        layer_id=job.layer_id,
                        scope=ArtifactScope.LAYER_ATTACHMENT,
                        name=filename,
                    )
                    if aggregate_output
                    else ArtifactSeries.for_frame(
                        series_id=series_id,
                        project_id=project.id,
                        layer_id=job.layer_id,
                        representation_id=target.id,
                        frame_id=result.frame_id,
                        name=filename,
                    )
                )
                if new_series:
                    series = expected_series
                elif (
                    series.archived
                    or series.project_id != project.id
                    or series.layer_id != job.layer_id
                    or series.scope != expected_series.scope
                    or series.representation_id != expected_series.representation_id
                    or series.frame_id != expected_series.frame_id
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
                coordinate = None if aggregate_output else coordinates_by_frame.get(result.frame_id)
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
                    input_version_ids=(
                        tuple(item.artifact_version_id for item in original.inputs)
                        if aggregate_output
                        else (input_item.artifact_version_id,)
                    ),
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
