"""composition: plugin jobs."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from kraken_manager.domain.artifacts import (
    ArtifactVersion,
)
from kraken_manager.domain.common import (
    ArtifactVersionId,
    ProjectId,
)
from kraken_manager.domain.workflows import (
    ReviewBatch,
)
from kraken_manager.infrastructure.filesystem import (
    FilesystemEventStore,
)
from kraken_manager.infrastructure.blob.filesystem import FilesystemBlobStore
from kraken_manager.infrastructure.filesystem.event_store import StoredEvent
from kraken_manager.infrastructure.filesystem.history import GenericHistoryEvent
from kraken_manager.infrastructure.reports import ActivityRecord


class PluginJobsMixin:
    def read_project_blob(self, project_id: ProjectId | str, sha256: str) -> bytes:
        return FilesystemBlobStore.for_project(self.catalog_root, str(project_id)).read(sha256)

    def get_artifact_version(
        self,
        project_id: ProjectId | str,
        artifact_version_id: ArtifactVersionId | str,
    ) -> ArtifactVersion | None:
        return self._projection(project_id).get_artifact_version(ArtifactVersionId(str(artifact_version_id)))

    def artifact_source_path(
        self,
        project_id: ProjectId | str,
        artifact_version_id: ArtifactVersionId | str,
    ) -> Path:
        """Resolve a managed artifact for read-only staging into the local Agent."""

        version = self.get_artifact_version(project_id, artifact_version_id)
        if version is None:
            raise KeyError(f"Unknown artifact version: {artifact_version_id}")
        if version.blob is None:
            raise ValueError("Analysis v1 requires managed model outputs in Kraken BlobStore")
        blob_store = FilesystemBlobStore.for_project(self.catalog_root, str(project_id))
        blob_store.verify(version.blob)
        return blob_store.path_for_read(version.blob)

    def history(self, project_id: ProjectId | str, *, as_of: datetime | None = None) -> tuple[object, ...]:
        store = FilesystemEventStore(self.catalog_root, str(project_id))
        events = []
        for stored in store.iter_project():
            event = store._decode(stored)
            if isinstance(event, StoredEvent):
                event = GenericHistoryEvent.from_stored(event)
            if as_of is None or event.recorded_at <= as_of:
                events.append(event)
        return tuple(events)

    def activity_records(self) -> tuple[ActivityRecord, ...]:
        records = [
            ActivityRecord.from_event(event)
            for project in self.list_projects(include_archived=True)
            for event in self.history(project.id)
        ]
        return tuple(sorted(records, key=lambda item: (item.recorded_at, item.event_id)))

    def active_review_batches(self) -> tuple[ReviewBatch, ...]:
        batches: dict[str, ReviewBatch] = {}
        for project in self.list_projects(include_archived=True):
            projection = self._projection(project.id)
            for layer in projection.list_layers(project.id, include_archived=True):
                for batch in projection.list_active_review_batches(project.id, layer.id):
                    batches[str(batch.id)] = batch
        return tuple(sorted(batches.values(), key=lambda item: (item.due_at is None, item.due_at, str(item.id))))

    def review_batches(self) -> tuple[ReviewBatch, ...]:
        batches: dict[str, ReviewBatch] = {}
        for project in self.list_projects(include_archived=True):
            projection = self._projection(project.id)
            for batch in projection.list_review_batches(project.id):
                batches[str(batch.id)] = batch
        return tuple(
            sorted(
                batches.values(),
                key=lambda item: (item.updated_at, str(item.id)),
                reverse=True,
            )
        )

    def plugin_jobs(self) -> tuple[object, ...]:
        jobs: dict[str, object] = {}
        for project in self.list_projects(include_archived=True):
            removed = {
                str(event.payload.get("plugin_job_id", ""))
                for event in self.history(project.id)
                if event.event_type == "LayerPipelineActionRemoved" and event.payload.get("remove_task")
            }
            for job in self._projection(project.id).list_plugin_jobs(project.id):
                if str(job.id) not in removed:
                    jobs[str(job.id)] = job
        return tuple(
            sorted(
                jobs.values(),
                key=lambda item: (getattr(item, "updated_at", None), str(item.id)),
                reverse=True,
            )
        )
