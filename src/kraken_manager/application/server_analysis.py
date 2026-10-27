"""Canonical server AnalysisRun lifecycle on the existing project EventStore."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import UUID, uuid5
import re

from kraken_core.analysis_protocol import AnalysisArtifactInput, AnalysisFrameInput, AnalysisSourceRole
from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest, AnalysisSourceBinding, payload_sha256
from kraken_manager.application.errors import ConflictError, ConcurrencyError, NotFoundError
from kraken_manager.application.pipeline_execution import deepcopy
from kraken_manager.domain.artifacts import deterministic_frame_series_id
from kraken_manager.domain.common import validate_uuid
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.project import RepresentationKind, StructureState


def resolve_analysis_inputs(projections, project, layer_id, coordinates, bindings):
    layer = projections.get_layer(validate_uuid(str(layer_id)))
    if layer is None or str(layer.project_id) != str(project.id) or layer.state is StructureState.ARCHIVED:
        raise NotFoundError("Active analysis layer was not found in the project")
    if not isinstance(bindings, Mapping) or not bindings or len(bindings) > 16:
        raise ConflictError("Analysis requires 1..16 named source bindings")
    representations = {}
    for key, identifier in bindings.items():
        if not isinstance(key, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", key) is None:
            raise ConflictError("Analysis binding keys must be safe names")
        representation = projections.get_representation(validate_uuid(str(identifier)))
        if (
            representation is None
            or str(representation.project_id) != str(project.id)
            or str(representation.layer_id) != str(layer.id)
            or representation.state is StructureState.ARCHIVED
            or representation.kind is not RepresentationKind.IMAGE
        ):
            raise ConflictError("Analysis binding must be an active image representation in this layer")
        representations[key] = representation
    coordinates = tuple(tuple(pair) for pair in coordinates)
    if not coordinates or len(coordinates) != len(set(coordinates)):
        raise ConflictError("Analysis coordinates must be nonempty and unique")
    frames, digests = [], {key: [] for key in bindings}
    for x, y in sorted(coordinates, key=lambda pair: (pair[1], pair[0])):
        frame_id = project.frame_id_at(x, y)
        artifacts = []
        for key in sorted(representations):
            representation = representations[key]
            series_id = deterministic_frame_series_id(representation.id, frame_id)
            series = projections.get_artifact_series(series_id)
            version = projections.get_active_artifact_version(series_id)
            if (
                series is None
                or series.archived
                or version is None
                or version.blob is None
                or not version.media_type.startswith("image/")
            ):
                raise ConflictError(f"Analysis needs a managed image for {key} at ({x}, {y})")
            suffix = version.filename.rsplit(".", 1)[-1].lower()
            if suffix not in {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}:
                raise ConflictError("Unsupported analysis image filename")
            artifacts.append(
                AnalysisArtifactInput(
                    key,
                    AnalysisSourceRole.MODEL_OUTPUT,
                    str(series_id),
                    str(version.id),
                    f"inputs/{key}/{frame_id}.{suffix}",
                    version.media_type,
                    version.sha256,
                    display_name=representation.name,
                )
            )
            digests[key].append(version.sha256)
        frames.append(AnalysisFrameInput(str(frame_id), x, y, tuple(artifacts)))
    sources = tuple(
        AnalysisSourceBinding(
            key, str(representations[key].id), payload_sha256(digests[key]), representations[key].name
        )
        for key in sorted(representations)
    )
    return tuple(frames), sources


class ServerAnalysisWorkflow:
    def __init__(self, events):
        self.events = events

    @staticmethod
    def stream(run_id):
        return f"analysis-run:{validate_uuid(str(run_id))}"

    def load(self, run_id):
        events = self.events.load_stream(self.stream(run_id))
        state = None
        for event in events:
            payload = deepcopy(event.payload)
            if event.event_type == "analysis_run.created":
                state = {
                    "manifest": payload["manifest"],
                    "layer_id": payload["layer_id"],
                    "actor_id": payload["actor_id"],
                    "resources": payload["resources"],
                    "cancel_jobs": [],
                    "maps": {},
                    "worker_runtime": None,
                    "partitions": {
                        p["partition_id"]: {"manifest": p, "state": "queued", "error": ""}
                        for p in payload["partitions"]
                    },
                }
            elif state is not None and event.event_type.startswith("analysis_map."):
                frame_id = payload["frame_id"]
                if event.event_type == "analysis_map.requested":
                    state["maps"][frame_id] = {**payload, "state": "queued"}
                else:
                    entry = state["maps"][frame_id]
                    if entry["manifest"]["job_id"] != payload["job_id"]:
                        raise ConflictError("Map history refers to a superseded attempt")
                    entry.update(payload)
                    entry["state"] = {"analysis_map.dispatched": "running", "analysis_map.imported": "ready",
                                      "analysis_map.failed": "failed"}[event.event_type]
            elif state is not None and event.event_type == "analysis_run.command":
                state["cancel_jobs"].extend(
                    identifier
                    for identifier in payload.get("cancel_jobs", ())
                    if identifier not in state["cancel_jobs"]
                )
                if payload["operation"] == "cancel":
                    for entry in state["maps"].values():
                        if entry["state"] in {"queued", "running"}:
                            entry["state"] = "cancelled"
                    for partition in state["partitions"].values():
                        if partition["state"] in {"queued", "running"}:
                            partition["state"] = "cancelled"
                for replacement in payload.get("replacements", ()):
                    state["partitions"][replacement["partition_id"]] = {
                        "manifest": replacement,
                        "state": "queued",
                        "error": "",
                    }
            elif state is not None and event.event_type in {
                "analysis_partition.dispatched",
                "analysis_partition.failed",
                "analysis_partition.imported",
            }:
                partition = state["partitions"][payload["partition_id"]]
                if partition["manifest"]["job_id"] != payload["job_id"]:
                    raise ConflictError("Analysis history refers to a superseded partition attempt")
                if event.event_type == "analysis_partition.imported":
                    from kraken_core.analysis_run_protocol import AnalysisPartitionResultManifest, AnalysisRuntimeIdentity
                    from kraken_core.analysis_runtime import validate_worker_runtime
                    result = AnalysisPartitionResultManifest.from_payload(payload["result"])
                    validate_worker_runtime(AnalysisPartitionJobManifest.from_payload(partition["manifest"]), result,
                        pinned=None if state["worker_runtime"] is None else AnalysisRuntimeIdentity.from_payload(state["worker_runtime"]))
                    if result.worker_runtime is not None:
                        state["worker_runtime"] = result.worker_runtime.to_payload()
                partition.update(payload)
                partition["state"] = {
                    "analysis_partition.dispatched": "running",
                    "analysis_partition.failed": "failed",
                    "analysis_partition.imported": "imported",
                }[event.event_type]
        if state is None:
            raise NotFoundError("Analysis run was not found")
        state["revision"] = events[-1].revision
        statuses = {p["state"] for p in state["partitions"].values()}
        failed_frames = sum(p.get("failed_frames", 0) for p in state["partitions"].values())
        ready_frames = sum(p.get("completed_frames", 0) for p in state["partitions"].values())
        state["state"] = (
            (("partial" if ready_frames else "failed") if failed_frames else "completed")
            if statuses == {"imported"}
            else (
                "running"
                if "running" in statuses
                else "queued"
                if "queued" in statuses
                else "partial"
                if "imported" in statuses
                else "failed"
                if "failed" in statuses
                else "cancelled"
            )
        )
        return state

    def append(self, run_id, project_id, event_type, payload, *, principal, expected_revision, key, command, at):
        if key:
            for prior in self.events.find_by_idempotency_key(project_id, key):
                if (
                    prior.stream_id != self.stream(run_id)
                    or prior.actor.principal_id != principal.id
                    or deepcopy(prior.payload.get("command")) != deepcopy(command)
                ):
                    raise ConflictError("Analysis idempotency key belongs to another command or actor")
                return False
        if self.events.current_revision(self.stream(run_id)) != expected_revision:
            raise ConcurrencyError("Analysis revision changed")
        event = EventEnvelope.create(
            stream_id=self.stream(run_id),
            project_id=project_id,
            revision=expected_revision + 1,
            event_type=event_type,
            payload={**payload, "command": command},
            actor=ActorSnapshot.from_principal(principal),
            recorded_at=at,
            idempotency_key=key,
        )
        self.events.append(event.stream_id, expected_revision=expected_revision, events=(event,))
        return True

    @staticmethod
    def command_id(project_id, key):
        return str(uuid5(UUID(str(project_id)), "analysis-run:" + key))
