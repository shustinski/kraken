"""Native layer ports/adaptation inside the existing pipeline unit of work."""
from dataclasses import replace
import hashlib

from kraken_core.plugin_protocol import PluginJobManifestV2
from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.application.multilayer import MultilayerService
from kraken_manager.application.quality import QualityService
from kraken_manager.application.stitch_content import content_versions
from kraken_manager.application.stitch_requests import request_stitch
from kraken_manager.application.stitch_results import _record
from kraken_manager.domain.artifacts import deterministic_frame_series_id
from kraken_manager.domain.project import RepresentationKind, RepresentationPurpose, StructureState


class PipelineStitchBindings:
    def __init__(self, bindings):
        self.bindings = bindings
        self.service = MultilayerService(bindings.events, bindings.projections)

    def geometry(self, op):
        model = self.service.load(str(self.bindings.project.id))
        stitch = next((s for s in model.stitches if (s.id, s.revision) == (op.stitch_id, op.stitch_revision)), None)
        if stitch is None or stitch.layer_id != self.bindings.layer_id:
            raise NotFoundError("Pipeline stitch geometry not found in this layer")
        return model, stitch

    def bind(self, op, producer_outputs):
        model, stitch = self.geometry(op)
        if not stitch.ready or not self.service.stitch_current(model, stitch):
            raise ConflictError("Pipeline stitch geometry is stale or unresolved")
        versions = {}
        if op.content is not None:
            if op.content.producer_id:
                for identifier in producer_outputs:
                    version = self.bindings.projections.get_artifact_version(identifier)
                    series = None if version is None else self.bindings.projections.get_artifact_series(version.series_id)
                    if series is not None and series.frame_id is not None:
                        versions[str(series.frame_id)] = str(version.id)
            else:
                for tile in stitch.tiles:
                    series_id = deterministic_frame_series_id(op.content.representation_id, tile.frame_id)
                    version = self.bindings.projections.get_active_artifact_version(series_id)
                    if version is None:
                        raise ConflictError("Layer content representation has a missing tile")
                    versions[tile.frame_id] = str(version.id)
        kind = "vector" if op.content is not None and op.content.kind.value == "vector" else "image"
        samples = None if op.content is None else content_versions(self.service, model, stitch, versions, kind=kind)
        anchors = [stitch.plan, *(tile.calibration.artifact for tile in stitch.tiles)]
        identifiers = {a.version_id for a in anchors} | (set(versions.values()) if versions else set())
        heads = {}
        for identifier in identifiers:
            version = self.bindings.projections.get_artifact_version(identifier)
            heads[str(version.series_id)] = str(version.id)
        if op.quality_gate:
            QualityService.require_clear_inputs(self.bindings.events, model.project_id, tuple(identifiers), projections=self.bindings.projections)
        parameters = dict(op.parameters)
        if op.content is not None and op.content.kind.value == "binary_mask":
            if parameters.get("mode", "labels") != "labels":
                raise ConflictError("Binary mask mosaics require nearest label-preserving sampling")
            parameters["mode"] = "labels"
        coordinates = {}
        for tile in stitch.tiles:
            original = self.bindings.projections.get_artifact_version(tile.calibration.artifact.version_id)
            sampled = original if samples is None else samples[tile.frame_id]
            coordinates[tile.frame_id] = tile.catalog_xy or (
                original.parameters.get("x", sampled.parameters.get("x")), original.parameters.get("y", sampled.parameters.get("y")))
        return {"kind": "layer", "layer_id": stitch.layer_id, "capability": op.capability,
            "stitch_id": stitch.id, "stitch_revision": stitch.revision, "parameters": parameters,
            "source_versions": versions if op.content is not None else None, "frame_coordinates": coordinates,
            **({"content_kind": "vector"} if kind == "vector" else {}),
            "input_heads": heads}

    def validate(self, op):
        self.geometry(op)
        if op.content is None or op.content.representation_id is None:
            return
        representation = self.bindings.projections.get_representation(op.content.representation_id)
        if (representation is None or str(representation.project_id) != str(self.bindings.project.id)
                or str(representation.layer_id) != self.bindings.layer_id or representation.state is StructureState.ARCHIVED
                or representation.kind is not (RepresentationKind.VECTOR if op.content.kind.value == "vector" else RepresentationKind.IMAGE)):
            raise ConflictError("Layer content representation does not belong to this active project layer")
        expected = {"binary_mask": RepresentationPurpose.BINARY, "confidence": RepresentationPurpose.CONFIDENCE}.get(op.content.kind.value)
        if expected is not None and representation.purpose is not expected:
            raise ConflictError("Layer content representation has the wrong typed purpose")

    def current(self, request):
        from types import SimpleNamespace
        model, stitch = self.geometry(SimpleNamespace(stitch_id=request["stitch_id"], stitch_revision=request["stitch_revision"]))
        return self.service.stitch_current(model, stitch) and stitch.ready

    def observe(self, job_id):
        model = self.service.load(str(self.bindings.project.id))
        request = next((r for r in model.stitch_renders if r.job_id == job_id), None)
        if request is None:
            return None
        manifest = PluginJobManifestV2.from_json(request.manifest_json)
        owner = manifest.parameters.get("_pipeline")
        if owner is None or manifest.layer_id != self.bindings.layer_id:
            raise ConflictError("Pipeline cannot adopt an unrelated stitch request")
        outputs = []
        state, error = request.state, request.error
        if state == "imported":
            version = self.bindings.projections.get_artifact_version(request.output.version_id)
            series = None if version is None else self.bindings.projections.get_artifact_series(version.series_id)
            if (series is None or str(series.project_id) != model.project_id or str(series.layer_id) != self.bindings.layer_id
                    or series.scope.value != "layer_attachment" or version.sha256 != request.output.sha256
                    or version.tool_version != owner["algorithm_version"] or not request.receipt_sha256):
                state, error = "failed", "Published mosaic does not match the pipeline output contract"
            else:
                state, outputs = "succeeded", [str(version.id)]
        return {"state": "queued" if state == "requested" else state, "error": error, "output_version_ids": outputs}


def ensure_stitch_intent(unit, bindings, intent, principal):
    request = intent["request"]
    service = MultilayerService(unit.event_store, unit.projections)
    model = service.load(str(bindings.project.id))
    existing = next((r for r in model.stitch_renders if r.job_id == intent["job_id"]), None)
    owner = request["parameters"]["_pipeline"]
    if existing is not None:
        manifest = PluginJobManifestV2.from_json(existing.manifest_json)
        wanted = request.get("source_versions") or {}
        sampled = {p["frame_id"]: p["artifact_version_id"] for p in manifest.parameters.get("content_inputs", ())}
        if (manifest.parameters.get("_pipeline") != owner or manifest.actor_id != str(principal.id)
                or manifest.created_at != intent["created_at"] or existing.stitch_id != request["stitch_id"]
                or existing.stitch_revision != request["stitch_revision"] or sampled != wanted
                or manifest.parameters["mode"] != request["parameters"].get("mode", "rgb")
                or manifest.parameters["block_size"] != request["parameters"].get("block_size", 256)
                or manifest.parameters.get("max_file_bytes") != request["parameters"].get("max_file_bytes")
                or manifest.parameters.get("content_kind", "image") != request.get("content_kind", "image")):
            raise ConflictError("Pipeline stitch job UUID already belongs to another request")
        return existing
    if not bindings.current(request):
        raise ConflictError("Pipeline stitch input changed before reservation")
    arguments = {"stitch_id": request["stitch_id"], "stitch_revision": request["stitch_revision"],
        "frame_coordinates": request["frame_coordinates"], **{k: v for k, v in request["parameters"].items() if k in {"mode", "block_size"}}}
    changed = request_stitch(service, model, arguments, intent["job_id"], str(principal.id),
        source_versions=request["source_versions"], pipeline_owner=owner, created_at=intent["created_at"],
        max_file_bytes=request["parameters"].get("max_file_bytes"), content_kind=request.get("content_kind", "image"))
    created = changed.stitch_renders[-1]
    _record(unit, replace(model, stitch_renders=changed.stitch_renders), created, principal,
        "pipeline_stitch_requested", hashlib.sha256(created.manifest_json.encode()).hexdigest())
    return created


def cancel_stitch_intent(unit, project_id, job_id, principal):
    service = MultilayerService(unit.event_store, unit.projections)
    model = service.load(str(project_id))
    request = next((r for r in model.stitch_renders if r.job_id == job_id), None)
    if request is None or request.state != "requested":
        return
    _record(unit, model, replace(request, state="cancelled"), principal, "pipeline_stitch_cancelled", hashlib.sha256(job_id.encode()).hexdigest())
