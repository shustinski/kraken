"""plugin_jobs: result payload."""
from __future__ import annotations
import hashlib
import json
from collections.abc import Mapping
from kraken_manager.application.errors import ConflictError
from kraken_manager.domain.workflows import (
    PluginJob,
    PluginJobState,
    PluginResultManifestV1,
)

from kraken_manager.application.plugin_jobs_parts.plain import _plain

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

def _move_job(job: PluginJob, target: PluginJobState, *, at: object, progress: float | None = None, error: str | None = None) -> PluginJob:
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
    return job.transition(target, at=timestamp, progress=progress, error=error)
