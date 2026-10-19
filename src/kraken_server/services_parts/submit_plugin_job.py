"""services: submit plugin job."""
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from kraken_manager.domain.common import LayerId, PrincipalId, ProjectId, RepresentationId
from kraken_manager.domain.selection import FrameRowRange, FrameSelectionV1
from kraken_manager.domain.workflows import PluginJob, PluginJobState
from kraken_manager.infrastructure.filesystem._codec import encode_model

from kraken_server.services_parts.command_context import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ValidationError,
)


class SubmitPluginJobMixin:
    def append_pipeline_event(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]:
        event_type = str(payload.get("event_type", ""))
        if event_type not in {"LayerPipelineActionRequested", "LayerPipelineActionRemoved"}:
            raise ValidationError("Unsupported pipeline event type")
        layer_id = str(payload.get("layer_id", ""))
        if not any(item["layer_id"] == layer_id for item in self._layers.get(project_id, ())):
            raise NotFoundError("Layer was not found")
        return self._append_auxiliary_event(
            project_id,
            stream_id=f"layer-pipeline:{layer_id}",
            event_type=event_type,
            payload={key: value for key, value in payload.items() if key != "event_type"},
            context=context,
        )

    def list_plugin_jobs(self, project_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            removed = {
                str(event.get("payload", {}).get("plugin_job_id", ""))
                for events in self._events.values()
                for event in events
                if event.get("event_type") == "LayerPipelineActionRemoved"
                and event.get("payload", {}).get("remove_task")
            }
            jobs = [
                dict(job)
                for job in self._plugin_jobs.values()
                if (project_id is None or str(job.get("project_id")) == project_id)
                and str(job.get("id")) not in removed
            ]
        jobs.sort(key=lambda item: (str(item.get("updated_at", "")), str(item.get("id", ""))), reverse=True)
        return jobs

    def authorize_desktop_job(self, project_id: str, principal_id: str) -> None:
        if not self.project_roles(project_id, principal_id).get("roles"):
            raise ForbiddenError("Project membership is required")

    def start_desktop_job(self, project_id: str, job_id: str, context: CommandContext) -> None:
        with self._lock:
            job = self._plugin_jobs.get(job_id)
            if job is None or str(job.get("project_id")) != project_id:
                raise NotFoundError("Job was not found")
            job["state"] = "running" if job["capability"] == "frames.binary-segment.v2" else "waiting_for_user"
            job["revision"] = int(job["revision"]) + 1

    def submit_plugin_job(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]:
        key = (f"plugin-job:{project_id}:{context.actor_id}", context.idempotency_key)
        with self._lock:
            if key in self._idempotency:
                return dict(self._idempotency[key])
            if project_id not in self._projects:
                raise NotFoundError(project_id)
            layer_id = str(payload.get("layer_id", "")).strip()
            target_id = str(payload.get("target_representation_id", "")).strip()
            capability = str(payload.get("capability", "")).strip()
            if not layer_id or not target_id or not capability:
                raise ValidationError(
                    "layer_id, target_representation_id and capability are required"
                )
            layers = self._layers.get(project_id, [])
            if layers and not any(str(item.get("layer_id")) == layer_id for item in layers):
                raise NotFoundError(layer_id)
            raw_coordinates = payload.get("coordinates", ())
            if not isinstance(raw_coordinates, list):
                raise ValidationError("coordinates must be an array")
            coordinates = sorted(
                {(int(item[0]), int(item[1])) for item in raw_coordinates},
                key=lambda item: (item[1], item[0]),
            )
            selection = FrameSelectionV1(
                row_ranges=tuple(
                    FrameRowRange(y=y, x_start=x, x_end=x) for x, y in coordinates
                )
            )
            job = PluginJob.create(
                project_id=ProjectId(project_id),
                layer_id=LayerId(layer_id),
                selection=selection,
                actor_principal_id=PrincipalId(context.actor_id),
                target_representation_id=RepresentationId(target_id),
                capability=capability,
            )
            encoded = encode_model(job)
            self._plugin_jobs[str(job.id)] = encoded
            self._idempotency[key] = encoded
            return dict(encoded)

    def cancel_plugin_job(
        self, project_id: str, job_id: str, context: CommandContext
    ) -> dict[str, Any]:
        key = (f"plugin-job-cancel:{job_id}:{context.actor_id}", context.idempotency_key)
        with self._lock:
            if key in self._idempotency:
                return dict(self._idempotency[key])
            job = self._plugin_jobs.get(job_id)
            if job is None or str(job.get("project_id")) != project_id:
                raise NotFoundError(job_id)
            if context.expected_revision is not None and int(job.get("revision", -1)) != context.expected_revision:
                raise ConflictError("Plugin job revision changed")
            if str(job.get("state")) in {
                PluginJobState.SUCCEEDED.value,
                PluginJobState.FAILED.value,
                PluginJobState.CANCELLED.value,
            }:
                raise ConflictError("Plugin job is already terminal")
            now = datetime.now(UTC).isoformat()
            updated = dict(job)
            updated["state"] = PluginJobState.CANCELLED.value
            updated["revision"] = int(job.get("revision", 0)) + 1
            updated["updated_at"] = now
            updated["finished_at"] = now
            self._plugin_jobs[job_id] = updated
            self._idempotency[key] = updated
            return dict(updated)

    def fail_agent_job(self, job_id: str, error: str) -> dict[str, Any]:
        with self._lock:
            job = self._plugin_jobs.get(job_id)
            if job is None:
                raise NotFoundError(job_id)
            now = datetime.now(UTC).isoformat()
            updated = dict(job)
            updated["state"] = PluginJobState.FAILED.value
            updated["error"] = str(error)
            updated["revision"] = int(job.get("revision", 0)) + 1
            updated["updated_at"] = now
            updated["finished_at"] = now
            self._plugin_jobs[job_id] = updated
            return dict(updated)

    def import_agent_result(self, job_id: str, agent: Any) -> dict[str, Any]:
        del agent
        with self._lock:
            job = self._plugin_jobs.get(job_id)
            if job is None:
                raise NotFoundError(job_id)
            now = datetime.now(UTC).isoformat()
            updated = dict(job)
            updated["state"] = PluginJobState.SUCCEEDED.value
            updated["progress"] = 1.0
            updated["revision"] = int(job.get("revision", 0)) + 1
            updated["updated_at"] = now
            updated["finished_at"] = now
            self._plugin_jobs[job_id] = updated
            return dict(updated)

if TYPE_CHECKING:
    from kraken_server.services_parts.command_context import CommandContext
