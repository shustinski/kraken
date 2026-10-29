"""Atomic layer artifact + physical-model receipt in the existing unit of work."""
from dataclasses import replace
import hashlib
from uuid import UUID, uuid5

from kraken_core.plugin_protocol import MOSAIC_MEDIA_TYPE, PluginJobManifestV2
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.domain.artifacts import ArtifactScope, ArtifactSeries, ArtifactVersion
from kraken_manager.domain.common import utc_now
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.multilayer import ArtifactAnchor
from kraken_manager.domain.project import ProjectState
from .multilayer import MultilayerService


def _context(unit, project_id, job_id, principal, storage, roles, verified):
    service = MultilayerService(unit.event_store, unit.projections)
    model = service.load(project_id)
    project = service._project(project_id)
    if project.state is not ProjectState.ACTIVE:
        raise ConflictError("Archived project is read-only")
    AuthorizationPolicy().require(principal=principal, storage=storage, roles=roles,
        permission=Permission.RUN_PLUGIN, gitlab_identity_verified=verified)
    request = next((r for r in model.stitch_renders if r.job_id == job_id), None)
    if request is None:
        raise NotFoundError("Stitch request not found")
    manifest = PluginJobManifestV2.from_json(request.manifest_json)
    if manifest.project_id != str(project_id) or manifest.job_id != job_id or manifest.operation != "layer.stitch.v1":
        raise ConflictError("Canonical stitch request identity mismatch")
    return service, model, request, manifest


def _record(unit, model, request, principal, operation, digest, *, actor_snapshot=None, program=None):
    changed = replace(model, revision=model.revision + 1, stitch_renders=tuple(
        request if r.job_id == request.job_id else r for r in model.stitch_renders))
    payload = changed.to_dict()
    payload.update(operation=operation, command_sha256=digest)
    event = EventEnvelope.create(stream_id=MultilayerService.stream(model.project_id), project_id=model.project_id,
        revision=changed.revision, event_type="MultilayerModelChangedV1", payload=payload,
        actor=actor_snapshot or ActorSnapshot.from_principal(principal), program=program,
        idempotency_key=f"{operation}:{request.job_id}")
    unit.event_store.append(event.stream_id, expected_revision=model.revision, events=(event,))
    return changed


def import_stitch_result(unit, *, project_id, principal, storage, roles, publication, source_path, validate_mosaic, open_source, verified=False):
    service, model, request, manifest = _context(unit, project_id, publication.job_id, principal, storage, roles, verified)
    digest = hashlib.sha256(publication.to_json().encode()).hexdigest()
    if request.state == "imported":
        if request.receipt_sha256 != digest:
            raise ConflictError("Another mosaic publication was already accepted")
        return model
    if request.state != "requested":
        raise ConflictError("Cancelled or failed stitch request cannot import a late result")
    from .stitch_pipeline import check_stitch_owner
    check_stitch_owner(unit.event_store, unit.projections, manifest)
    stitch = next(s for s in model.stitches if (s.id, s.revision) == (request.stitch_id, request.stitch_revision))
    if not service.stitch_current(model, stitch) or not stitch.ready:
        raise ConflictError("Stitch source or geometry is stale")
    if not publication.final or publication.outcome != "succeeded" or len(publication.outputs) != 1:
        raise ConflictError("Mosaic requires one complete successful publication")
    asset = publication.outputs[0]
    if asset.scope.value != "layer" or asset.role != "mosaic" or asset.media_type != MOSAIC_MEDIA_TYPE:
        raise ConflictError("Invalid mosaic output scope, role or media type")
    with open_source(source_path) as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != asset.sha256:
            raise ConflictError("Mosaic output SHA-256 mismatch")
        metadata = validate_mosaic(source_path, stitch, request.job_id)
        if (metadata["render"]["mode"] != manifest.parameters["mode"]
                or metadata["render"]["block_size"] != manifest.parameters["block_size"]):
            raise ConflictError("Mosaic render policy differs from the requested policy")
        if metadata.get("content_inputs", []) != manifest.parameters.get("content_inputs", []):
            raise ConflictError("Mosaic content differs from the pinned pipeline bindings")
        if ("vector_stitch" in metadata) != (manifest.parameters.get("content_kind") == "vector"):
            raise ConflictError("Mosaic vector content policy differs from the pinned request")
        stream.seek(0)
        stored = unit.blobs.put(iter(lambda: stream.read(1024 * 1024), b""), expected_sha256=asset.sha256)
    series_id = str(uuid5(UUID(request.job_id), "mosaic-series"))
    version_id = str(uuid5(UUID(request.job_id), "mosaic-version"))
    if unit.projections.get_artifact_series(series_id) is not None or unit.projections.get_artifact_version(version_id) is not None:
        raise ConflictError("Mosaic output identity already belongs to another artifact")
    now = utc_now()
    series = ArtifactSeries(series_id, model.project_id, ArtifactScope.LAYER_ATTACHMENT, "Mosaic", layer_id=stitch.layer_id)
    inputs = tuple(dict.fromkeys((stitch.plan.version_id, *(tile.calibration.artifact.version_id for tile in stitch.tiles),
                                 *(p["artifact_version_id"] for p in manifest.parameters.get("content_inputs", [])))))
    version = ArtifactVersion.managed(version_id=version_id, series_id=series.id, blob=stored.blob,
        media_type=MOSAIC_MEDIA_TYPE, filename="mosaic.sqlite3", author_principal_id=principal.id, created_at=now,
        input_version_ids=inputs, tool_name="Kraken Stitch", tool_version=publication.plugin_version,
        parameters={"plugin_job_id": request.job_id, "plugin_output_role": "mosaic", "stitch_id": stitch.id,
                    "stitch_revision": stitch.revision, "plan_sha256": metadata["plan_sha256"], "render": metadata["render"],
                    **({"vector_stitch": metadata["vector_stitch"]} if "vector_stitch" in metadata else {})})
    stream_id = f"artifact-series:{series.id}"
    series_payload = {"artifact_series_id": series.id, "scope": series.scope.value, "name": series.name,
                      "layer_id": series.layer_id, "representation_id": None, "frame_id": None, "archived": False}
    version_payload = {"artifact_version_id": version.id, "series_id": series.id, "layer_id": series.layer_id,
        "representation_id": None, "frame_id": None, "sha256": version.sha256, "size_bytes": version.size_bytes,
        "media_type": version.media_type, "filename": version.filename,
        "blob": {"sha256": version.sha256, "size_bytes": version.size_bytes}, "parent_version_id": None,
        "input_version_ids": list(inputs), "tool_name": version.tool_name, "tool_version": version.tool_version,
        "parameters": dict(version.parameters), "author_principal_id": principal.id, "created_at": now.isoformat(),
        "activated": True, "branched": False, "plugin_job_id": request.job_id}
    events = tuple(EventEnvelope.create(stream_id=stream_id, project_id=model.project_id, revision=i,
        event_type=kind, payload=payload, actor=ActorSnapshot.from_principal(principal), recorded_at=now)
        for i, (kind, payload) in enumerate((("ArtifactSeriesCreated", series_payload), ("ArtifactVersionCreated", version_payload)), 1))
    unit.event_store.append(stream_id, expected_revision=0, events=events)
    unit.projections.save_artifact_series(series)
    unit.projections.save_artifact_version(version, activate=True)
    request = replace(request, state="imported", output=ArtifactAnchor(version.id, version.sha256), receipt_sha256=digest)
    return _record(unit, model, request, principal, "import_layer_stitch", digest)


def fail_stitch_result(unit, *, project_id, job_id, principal, storage, roles, error, verified=False):
    _, model, request, _ = _context(unit, project_id, job_id, principal, storage, roles, verified)
    if request.state != "requested":
        return model
    request = replace(request, state="failed", error=str(error)[:10000])
    return _record(unit, model, request, principal, "fail_layer_stitch", hashlib.sha256(request.error.encode()).hexdigest())


def reject_stitch_authorization(unit, *, project_id, job_id, identities, storage):
    """Server-only terminal rejection; cannot import artifacts or revive a job.

    Uses the real current identity/ACL, and attributes the automatic failure to
    the scheduler program and the original request actor snapshot.
    """
    service = MultilayerService(unit.event_store, unit.projections)
    model = service.load(project_id)
    request = next((r for r in model.stitch_renders if r.job_id == job_id), None)
    if request is None:
        raise NotFoundError("Stitch request not found")
    if request.state != "requested":
        return None
    if service._project(project_id).state is not ProjectState.ACTIVE:
        raise ConflictError("Archived project is read-only")
    manifest = PluginJobManifestV2.from_json(request.manifest_json)
    principal = identities.get(manifest.actor_id)
    decision = None if principal is None else AuthorizationPolicy().decide(principal=principal, storage=storage,
        roles=unit.acl.roles_for(project_id, manifest.actor_id), permission=Permission.RUN_PLUGIN, gitlab_identity_verified=True)
    if decision is not None and decision.allowed:
        return None
    original = next((event.actor for event in unit.event_store.load_stream(service.stream(project_id))
        if event.actor is not None and str(event.actor.principal_id) == manifest.actor_id
        and any(r["job_id"] == job_id for r in event.payload["model"].get("stitch_renders", ()))), None)
    if original is None:
        raise ConflictError("Stitch request actor snapshot is missing")
    error = "Initiating identity no longer exists" if decision is None else decision.reason
    request = replace(request, state="failed", error="Authorization lost: " + error)
    return _record(unit, model, request, principal, "reject_layer_stitch_authorization",
        hashlib.sha256(request.error.encode()).hexdigest(), actor_snapshot=original,
        program=ProgramSnapshot("Kraken Server scheduler", "1"))
