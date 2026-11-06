"""Retire active outputs of a removed step while retaining immutable history."""

from kraken_manager.application.ports import UnitOfWork
from kraken_manager.domain.common import ArtifactVersionId, PluginJobId, LayerId
from kraken_manager.domain.events import EventEnvelope


def removed_output_version_ids(uow: UnitOfWork, layer_id: LayerId) -> set[ArtifactVersionId]:
    """Also honor removal receipts written before output archival was introduced."""
    removed_jobs = {
        str(event.payload["plugin_job_id"])
        for event in uow.event_store.load_stream(f"layer-pipeline:{layer_id}")
        if event.event_type == "LayerPipelineActionRemoved" and event.payload.get("plugin_job_id")
    }
    return {
        ArtifactVersionId(str(identifier))
        for job_id in removed_jobs
        for event in uow.event_store.load_stream(f"plugin-job:{job_id}")
        if event.event_type == "PluginResultImported"
        for identifier in event.payload.get("artifact_version_ids", ())
    }


def archive_removed_job_outputs(uow: UnitOfWork, removal: EventEnvelope) -> None:
    job_id = str(removal.payload.get("plugin_job_id") or "")
    if not job_id:
        return
    job = uow.projections.get_plugin_job(PluginJobId(job_id))
    if job is None or job.project_id != removal.project_id or str(job.layer_id) != str(removal.payload["layer_id"]):
        raise ValueError("Removed job does not belong to this layer")
    identifiers = {
        ArtifactVersionId(str(identifier))
        for event in uow.event_store.load_stream(f"plugin-job:{job_id}")
        if event.event_type == "PluginResultImported"
        for identifier in event.payload.get("artifact_version_ids", ())
    }
    for identifier in sorted(identifiers):
        version = uow.projections.get_artifact_version(identifier)
        if version is None:
            raise ValueError("Imported output projection is missing")
        series = uow.projections.get_artifact_series(version.series_id)
        if series is None or series.project_id != job.project_id or series.layer_id != job.layer_id:
            raise ValueError("Imported output does not belong to the removed job layer")
        active = uow.projections.get_active_artifact_version(series.id)
        # A later operation can share a legacy series. Keep its newer output.
        if series.archived or active is None or active.id != identifier:
            continue
        archived = series.archive()
        stream = f"artifact-series:{series.id}"
        revision = uow.event_store.current_revision(stream)
        event = EventEnvelope.create(
            stream_id=stream,
            project_id=job.project_id,
            revision=revision + 1,
            event_type="ArtifactSeriesArchived",
            payload={
                "artifact_series_id": str(series.id),
                "archived": True,
                "series_revision": archived.revision,
                "plugin_job_id": job_id,
            },
            actor=removal.actor,
            program=removal.program,
            recorded_at=removal.recorded_at,
            correlation_id=removal.event_id,
        )
        uow.event_store.append(stream, expected_revision=revision, events=(event,))
        uow.projections.save_artifact_series(archived)
