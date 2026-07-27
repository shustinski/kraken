"""Recreate typed current/temporal projections from authoritative events."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Mapping
from datetime import datetime
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from kraken_manager.domain.artifacts import ArtifactSeries, ArtifactVersion, BlobRef
from kraken_manager.domain.common import ArtifactSeriesId, ArtifactVersionId, FrameId, LayerId, PerformerId, PluginJobId, PrincipalId, ProjectId, RepresentationId, ReviewBatchId
from kraken_manager.domain.events import EventEnvelope
from kraken_manager.domain.project import (
    GridOrientation,
    Layer,
    LayerType,
    Project,
    ProjectState,
    Representation,
    RepresentationKind,
    StructureState,
)
from kraken_manager.domain.artifacts import ArtifactScope
from kraken_manager.domain.identity import ProjectRole, ProjectRoleAssignment
from kraken_manager.domain.selection import FrameSelectionV1
from kraken_manager.domain.workflows import PluginJob, PluginJobState, ReviewBatch, ReviewBatchState, ReviewItem
from kraken_manager.infrastructure.filesystem._atomic import fsync_directory


Upcaster = Callable[[dict[str, object]], dict[str, object]]


class EventUpcasterRegistry:
    def __init__(self) -> None:
        self._upcasters: dict[tuple[str, int], Upcaster] = {}

    def register(self, event_type: str, from_version: int, upcaster: Upcaster) -> None:
        key = (event_type, from_version)
        if key in self._upcasters:
            raise ValueError(f"An upcaster is already registered for {event_type} v{from_version}")
        self._upcasters[key] = upcaster

    def upcast(self, event: EventEnvelope, *, target_version: int = 1) -> EventEnvelope:
        current = event
        while current.schema_version < target_version:
            upcaster = self._upcasters.get((current.event_type, current.schema_version))
            if upcaster is None:
                raise ValueError(f"Missing upcaster for {current.event_type} v{current.schema_version}")
            current = replace(
                current,
                payload=upcaster(dict(current.payload)),
                schema_version=current.schema_version + 1,
            )
        if current.schema_version > target_version:
            raise ValueError(f"Event {current.event_type} uses unsupported future schema v{current.schema_version}")
        return current


class ProjectionRebuilder:
    """Apply domain events to any typed projection adapter in recorded order."""

    def __init__(
        self,
        store: Any,
        *,
        upcasters: EventUpcasterRegistry | None = None,
        acl: Any | None = None,
    ) -> None:
        self.store = store
        self.upcasters = upcasters or EventUpcasterRegistry()
        self.acl = acl

    @staticmethod
    def _time(payload: dict[str, object], field: str, fallback: datetime) -> datetime:
        value = payload.get(field)
        if value is None:
            return fallback
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    def _save(self, kind: str, model: Any, event: EventEnvelope, *, active: bool | None = None) -> None:
        if hasattr(self.store, "_save_typed"):
            self.store._save_typed(kind, model, active=active, recorded_at=event.recorded_at)
            return
        if hasattr(self.store, "_save"):
            self.store._save(kind, model, active=active, recorded_at=event.recorded_at)
            return
        method = getattr(self.store, f"save_{kind}")
        if kind == "artifact_version":
            method(model, activate=bool(active))
        else:
            method(model)

    def apply(self, envelope: EventEnvelope) -> bool:
        event = self.upcasters.upcast(envelope)
        payload = dict(event.payload)
        if event.event_type == "ProjectCreated":
            project = Project(
                id=ProjectId(str(payload["project_id"])),
                name=str(payload["name"]),
                width=int(payload["width"]),
                height=int(payload["height"]),
                orientation=GridOrientation(str(payload["orientation"])),
                storage_profile=str(payload["storage_profile"]),
                state=ProjectState(str(payload.get("state", "active"))),
                revision=event.revision,
                created_at=self._time(payload, "created_at", event.recorded_at),
            )
            self._save("project", project, event)
            if self.acl is not None:
                self.acl.assign(
                    ProjectRoleAssignment.create(
                        project_id=project.id,
                        principal_id=event.actor.principal_id,
                        role=ProjectRole.OWNER,
                        assigned_by=event.actor.principal_id,
                        assigned_at=event.recorded_at,
                    )
                )
            return True
        if event.event_type == "ProjectRoleAssigned":
            if self.acl is not None:
                self.acl.assign(
                    ProjectRoleAssignment.create(
                        project_id=event.project_id,
                        principal_id=PrincipalId(str(payload["principal_id"])),
                        role=ProjectRole(str(payload["role"])),
                        assigned_by=PrincipalId(str(payload.get("assigned_by", event.actor.principal_id))),
                        assigned_at=self._time(payload, "changed_at", event.recorded_at),
                    )
                )
            return True
        if event.event_type == "ProjectRoleRevoked":
            if self.acl is not None:
                self.acl.revoke(
                    event.project_id,
                    PrincipalId(str(payload["principal_id"])),
                    ProjectRole(str(payload["role"])),
                )
            return True
        if event.event_type in {"ProjectRenamed", "ProjectArchived", "ProjectRestored"}:
            raw = payload.get("project")
            if not isinstance(raw, Mapping):
                raise ValueError("Project lifecycle event has no aggregate snapshot")
            project = Project(
                id=ProjectId(str(raw["id"])),
                name=str(raw["name"]),
                width=int(raw["width"]),
                height=int(raw["height"]),
                orientation=GridOrientation(str(raw["orientation"])),
                storage_profile=str(raw["storage_profile"]),
                state=ProjectState(str(raw["state"])),
                revision=event.revision,
                created_at=self._time(raw, "created_at", event.recorded_at),
            )
            self._save("project", project, event)
            return True
        if event.event_type == "LayerCreated":
            layer = Layer(
                id=LayerId(str(payload["layer_id"])),
                project_id=event.project_id,
                name=str(payload["name"]),
                type=LayerType(str(payload["type"])),
                order=int(payload["order"]),
                state=StructureState(str(payload.get("state", "active"))),
                revision=0,
                created_at=self._time(payload, "created_at", event.recorded_at),
            )
            self._save("layer", layer, event)
            project = self.store.get_project(event.project_id)
            if project is not None:
                self._save("project", replace(project, revision=event.revision), event)
            return True
        if event.event_type in {"LayerRenamed", "LayerReordered", "LayerArchived"}:
            raw = payload.get("layer")
            if not isinstance(raw, Mapping):
                raise ValueError("Layer lifecycle event has no aggregate snapshot")
            layer = Layer(
                id=LayerId(str(raw["id"])),
                project_id=ProjectId(str(raw["project_id"])),
                name=str(raw["name"]),
                type=LayerType(str(raw["type"])),
                order=int(raw["order"]),
                state=StructureState(str(raw["state"])),
                revision=event.revision,
                created_at=self._time(raw, "created_at", event.recorded_at),
            )
            self._save("layer", layer, event)
            return True
        if event.event_type == "RepresentationCreated":
            representation = Representation(
                id=RepresentationId(str(payload["representation_id"])),
                project_id=event.project_id,
                layer_id=LayerId(str(payload["layer_id"])),
                name=str(payload["name"]),
                kind=RepresentationKind(str(payload["kind"])),
                note=str(payload.get("note", "")),
                source=None if payload.get("source") is None else str(payload["source"]),
                source_image_representation_id=(
                    None
                    if payload.get("source_image_representation_id") is None
                    else RepresentationId(str(payload["source_image_representation_id"]))
                ),
                active=bool(payload.get("active", False)),
                state=StructureState(str(payload.get("state", "active"))),
                revision=0,
                created_at=self._time(payload, "created_at", event.recorded_at),
            )
            for identifier in payload.get("deactivated_representation_ids", ()):
                previous = self.store.get_representation(RepresentationId(str(identifier)))
                if previous is not None and previous.active:
                    self._save("representation", previous.deactivate(), event)
            self._save("representation", representation, event)
            layer = self.store.get_layer(representation.layer_id)
            if layer is not None:
                self._save("layer", replace(layer, revision=event.revision), event)
            return True
        if event.event_type in {
            "RepresentationRenamed",
            "RepresentationNoteUpdated",
            "RepresentationActivated",
            "RepresentationArchived",
        }:
            raw = payload.get("representation")
            if not isinstance(raw, Mapping):
                raise ValueError("Representation lifecycle event has no aggregate snapshot")

            def representation_from_snapshot(item: Mapping[str, Any]) -> Representation:
                return Representation(
                    id=RepresentationId(str(item["id"])),
                    project_id=ProjectId(str(item["project_id"])),
                    layer_id=LayerId(str(item["layer_id"])),
                    name=str(item["name"]),
                    kind=RepresentationKind(str(item["kind"])),
                    note=str(item.get("note", "")),
                    source=None if item.get("source") is None else str(item["source"]),
                    source_image_representation_id=(
                        None
                        if item.get("source_image_representation_id") is None
                        else RepresentationId(str(item["source_image_representation_id"]))
                    ),
                    active=bool(item.get("active", False)),
                    state=StructureState(str(item["state"])),
                    revision=int(item["revision"]),
                    created_at=self._time(item, "created_at", event.recorded_at),
                )

            for previous in payload.get("deactivated", ()):
                if isinstance(previous, Mapping):
                    self._save("representation", representation_from_snapshot(previous), event)
            value = representation_from_snapshot(raw)
            self._save("representation", value, event)
            layer = self.store.get_layer(value.layer_id)
            if layer is not None:
                self._save("layer", replace(layer, revision=event.revision), event)
            return True
        if event.event_type == "ArtifactSeriesCreated":
            series = ArtifactSeries(
                id=ArtifactSeriesId(str(payload["artifact_series_id"])),
                project_id=event.project_id,
                scope=ArtifactScope(str(payload["scope"])),
                name=str(payload["name"]),
                layer_id=None if payload.get("layer_id") is None else LayerId(str(payload["layer_id"])),
                representation_id=None
                if payload.get("representation_id") is None
                else RepresentationId(str(payload["representation_id"])),
                frame_id=None if payload.get("frame_id") is None else FrameId(str(payload["frame_id"])),
                archived=bool(payload.get("archived", False)),
            )
            self._save("artifact_series", series, event)
            return True
        if event.event_type == "ArtifactVersionCreated":
            blob_payload = payload.get("blob")
            if not isinstance(blob_payload, Mapping):
                raise ValueError("Managed artifact event has no blob reference")
            blob = BlobRef(str(blob_payload["sha256"]), int(blob_payload["size_bytes"]))
            version = ArtifactVersion.managed(
                version_id=ArtifactVersionId(str(payload["artifact_version_id"])),
                series_id=ArtifactSeriesId(str(payload["series_id"])),
                blob=blob,
                media_type=str(payload["media_type"]),
                filename=str(payload["filename"]),
                author_principal_id=PrincipalId(str(payload.get("author_principal_id", event.actor.principal_id))),
                created_at=self._time(payload, "created_at", event.recorded_at),
                parent_version_id=None
                if payload.get("parent_version_id") is None
                else ArtifactVersionId(str(payload["parent_version_id"])),
                input_version_ids=tuple(ArtifactVersionId(str(value)) for value in payload.get("input_version_ids", ())),
                tool_name=None if payload.get("tool_name") is None else str(payload["tool_name"]),
                tool_version=None if payload.get("tool_version") is None else str(payload["tool_version"]),
                parameters=dict(payload.get("parameters", {})),
            )
            self._save("artifact_version", version, event, active=bool(payload.get("activated", False)))
            return True
        if event.event_type == "ArtifactVersionActivated":
            version = self.store.get_artifact_version(
                ArtifactVersionId(str(payload["artifact_version_id"]))
            )
            if version is None:
                raise ValueError("Artifact activation references an unknown version")
            self._save("artifact_version", version, event, active=True)
            return True
        if event.event_type in {
            "PluginJobCreated",
            "PluginResultAwaitingAuthorization",
            "PluginPartialResultReceived",
            "PluginResultImported",
            "PluginJobFailed",
            "PluginJobCancelled",
        }:
            raw_job = payload.get("job")
            if not isinstance(raw_job, Mapping):
                raise ValueError("Plugin job event has no aggregate snapshot")
            raw_selection = raw_job.get("selection")
            if not isinstance(raw_selection, Mapping):
                raise ValueError("Plugin job event has no selection snapshot")
            job_payload = dict(raw_job)
            job = PluginJob(
                id=PluginJobId(str(raw_job["id"])),
                project_id=ProjectId(str(raw_job["project_id"])),
                layer_id=LayerId(str(raw_job["layer_id"])),
                selection=FrameSelectionV1.from_dict(raw_selection),
                actor_principal_id=PrincipalId(str(raw_job["actor_principal_id"])),
                target_representation_id=RepresentationId(str(raw_job["target_representation_id"])),
                capability=str(raw_job["capability"]),
                state=PluginJobState(str(raw_job["state"])),
                revision=int(raw_job["revision"]),
                progress=float(raw_job["progress"]),
                created_at=self._time(job_payload, "created_at", event.recorded_at),
                updated_at=self._time(job_payload, "updated_at", event.recorded_at),
                finished_at=None
                if raw_job.get("finished_at") is None
                else self._time(job_payload, "finished_at", event.recorded_at),
                error=None if raw_job.get("error") is None else str(raw_job["error"]),
            )
            self._save("plugin_job", job, event)
            return True
        if event.event_type == "ReviewBatchCreated":
            selection_payload = payload.get("selection")
            if not isinstance(selection_payload, Mapping):
                raise ValueError("Review batch event has no frame selection")
            items_payload = payload.get("items")
            if not isinstance(items_payload, (tuple, list)):
                raise ValueError("Review batch event has no immutable items")
            batch = ReviewBatch(
                id=ReviewBatchId(str(payload["review_batch_id"])),
                project_id=event.project_id,
                layer_id=LayerId(str(payload["layer_id"])),
                selection=FrameSelectionV1.from_dict(selection_payload),
                items=tuple(
                    ReviewItem(
                        frame_id=FrameId(str(item["frame_id"])),
                        vector_version_id=ArtifactVersionId(str(item["vector_version_id"])),
                        vector_sha256=str(item["vector_sha256"]),
                        image_version_id=None
                        if item.get("image_version_id") is None
                        else ArtifactVersionId(str(item["image_version_id"])),
                    )
                    for item in items_payload
                    if isinstance(item, Mapping)
                ),
                assignee_id=PerformerId(str(payload["assignee_id"])),
                created_by=PrincipalId(str(payload["created_by"])),
                instructions=str(payload.get("instructions", "")),
                state=ReviewBatchState(str(payload.get("state", "draft"))),
                revision=int(payload.get("batch_revision", 0)),
                created_at=self._time(payload, "created_at", event.recorded_at),
                updated_at=self._time(payload, "updated_at", event.recorded_at),
                due_at=None
                if payload.get("due_at") is None
                else self._time(payload, "due_at", event.recorded_at),
            )
            self._save("review_batch", batch, event)
            layer = self.store.get_layer(batch.layer_id)
            if layer is not None:
                self._save("layer", replace(layer, revision=event.revision), event)
            return True
        if event.event_type in {
            "ReviewBatchIssued",
            "ReviewReturnCommitted",
            "ReviewBatchAccepted",
            "ReviewChangesRequested",
        }:
            batch = self.store.get_review_batch(ReviewBatchId(str(payload["review_batch_id"])))
            if batch is None:
                raise ValueError("Review transition references an unknown batch")
            transitioned = replace(
                batch,
                state=ReviewBatchState(str(payload["state"])),
                revision=int(payload.get("batch_revision", event.revision)),
                updated_at=self._time(payload, "updated_at", event.recorded_at),
            )
            self._save("review_batch", transitioned, event)
            return True
        return False

    def rebuild(self, events: Any) -> tuple[int, int]:
        applied = ignored = 0
        for event in events:
            if self.apply(event):
                applied += 1
            else:
                ignored += 1
        return applied, ignored


def rebuild_filesystem_index(
    event_store: Any,
    projection_store: Any,
    *,
    acl: Any | None = None,
) -> tuple[int, int]:
    """Build typed and event indexes beside the old SQLite file, then atomically swap."""

    index_path = Path(projection_store.path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="read-rebuild-", suffix=".sqlite3", dir=index_path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)
    rebuilt = object.__new__(type(projection_store))
    rebuilt.layout = projection_store.layout
    rebuilt.path = temporary
    rebuilt._ensure_schema()
    projector = ProjectionRebuilder(rebuilt, acl=acl)
    applied = ignored = 0
    try:
        for stored in event_store.iter_project():
            rebuilt.apply(stored)
            envelope = event_store._decode(stored)
            if not isinstance(envelope, EventEnvelope):
                ignored += 1
            elif projector.apply(envelope):
                applied += 1
            else:
                ignored += 1
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, index_path)
        fsync_directory(index_path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        Path(f"{temporary}-wal").unlink(missing_ok=True)
        Path(f"{temporary}-shm").unlink(missing_ok=True)
    return applied, ignored


__all__ = ["EventUpcasterRegistry", "ProjectionRebuilder", "rebuild_filesystem_index"]
