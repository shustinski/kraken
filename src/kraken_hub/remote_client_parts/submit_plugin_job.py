"""remote_client: submit plugin job."""
from __future__ import annotations
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from kraken_agent.runner import PluginRegistry
from kraken_manager.domain.common import validate_uuid
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.workflows import (
    PluginJob,
)
from kraken_manager.infrastructure.filesystem._codec import decode_model
from kraken_manager.infrastructure.reports import (
    ReportMetrics,
    ReportSeries,
)

from kraken_hub.remote_client_parts.extract_review_archive import RemoteServerError

from kraken_hub.remote_client_parts.remote_history_event import _RemoteHistoryEvent


class SubmitPluginJobMixin:
    def plugin_jobs(self) -> tuple[PluginJob, ...]:
        jobs: list[PluginJob] = []
        for project in self.list_projects(include_archived=True):
            payload = self._call("GET", f"/api/v1/projects/{project.id}/jobs")
            items = payload.get("items", ()) if isinstance(payload, Mapping) else ()
            jobs.extend(
                decode_model(PluginJob, item)
                for item in items
                if isinstance(item, Mapping)
            )
        return tuple(
            sorted(jobs, key=lambda item: (item.updated_at, str(item.id)), reverse=True)
        )

    def submit_plugin_job(
        self,
        *,
        principal: Principal,
        gateway: object,
        project_id: object,
        layer_id: object,
        source_representation_id: object,
        target_representation_id: object,
        coordinates: object,
        capability: str,
        parameters: Mapping[str, object],
        idempotency_key: str,
    ) -> PluginJob:
        del principal, gateway
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/jobs",
            payload={
                "layer_id": str(layer_id),
                "source_representation_id": str(source_representation_id),
                "target_representation_id": str(target_representation_id),
                "coordinates": [[int(x), int(y)] for x, y in coordinates],
                "capability": capability,
                "parameters": dict(parameters),
            },
            idempotency_key=idempotency_key,
        )
        return decode_model(PluginJob, payload)

    def run_interactive_job(self, project_id: object, job_id: object, *, registry: PluginRegistry) -> dict[str, Any]:
        from kraken_agent.server_worker import ServerAgentWorker

        job_id = validate_uuid(str(job_id), field="job_id")
        worker = ServerAgentWorker(
            server_url=self.base_url,
            token=self.auth.access_token,
            data_dir=self.data_dir / "desktop-jobs" / str(job_id),
            registry=registry,
            headers={"X-Kraken-Role": str(self.acting_role)} if getattr(self, "acting_role", None) else {},
        )
        lease = self._call("POST", f"/api/v1/projects/{project_id}/jobs/{job_id}/desktop-lease", payload={})
        if not isinstance(lease, Mapping) or not isinstance(lease.get("job"), dict):
            raise RemoteServerError("Server did not grant the interactive job lease")
        worker.token = self.auth.access_token
        return worker.run_leased_job(lease["job"])

    def cancel_plugin_job(
        self,
        *,
        principal: Principal,
        gateway: object,
        job: PluginJob,
        idempotency_key: str,
    ) -> PluginJob:
        del principal, gateway
        payload = self._call(
            "POST",
            f"/api/v1/projects/{job.project_id}/jobs/{job.id}/cancel",
            payload={},
            idempotency_key=idempotency_key,
            if_match=job.revision,
        )
        return decode_model(PluginJob, payload)

    def synchronize_plugin_jobs(
        self, *, principal: Principal, gateway: object
    ) -> tuple[PluginJob, ...]:
        del principal, gateway
        return self.plugin_jobs()

    def project_workspace(self, project_id: object):
        from kraken_manager.workspace import ProjectWorkspaceBinding

        try:
            payload = self._call("GET", f"/api/v1/projects/{project_id}/workspace")
        except RemoteServerError as exc:
            if exc.status == 404:
                return None
            raise
        if not isinstance(payload, Mapping):
            return None
        return ProjectWorkspaceBinding(**dict(payload))

    def project_storage_label(self, project_id: object) -> str:
        return "Сервер PostgreSQL"

    def is_remote_project(self, project_id: object) -> bool:
        return str(project_id) in self._remote_ids

    def history(
        self, project_id: object, *, as_of: datetime | None = None
    ) -> tuple[object, ...]:
        del as_of
        events = []
        cursor: str | None = None
        while True:
            query = urllib.parse.urlencode(
                {"limit": 500, **({} if cursor is None else {"cursor": cursor})}
            )
            payload = self._call(
                "GET", f"/api/v1/projects/{project_id}/history?{query}"
            )
            items = payload.get("items", []) if isinstance(payload, Mapping) else []
            events.extend(
                _RemoteHistoryEvent(item, project_id=str(project_id))
                for item in items
                if isinstance(item, Mapping)
            )
            cursor = (
                None
                if not isinstance(payload, Mapping)
                or payload.get("next_cursor") is None
                else str(payload["next_cursor"])
            )
            if cursor is None:
                break
        return tuple(events)

    def activity_records(self) -> tuple[object, ...]:
        from kraken_manager.infrastructure.reports import ActivityRecord

        records = [
            ActivityRecord.from_event(event)
            for project in self.list_projects(include_archived=True)
            for event in self.history(project.id)
        ]
        return tuple(
            sorted(records, key=lambda item: (item.recorded_at, item.event_id))
        )

    def statistics(
        self,
        project_id: object,
        *,
        start: datetime,
        end: datetime,
        timezone: object,
    ) -> tuple[ReportMetrics, dict[str, ReportSeries]]:
        query = urllib.parse.urlencode(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "timezone": str(getattr(timezone, "key", "UTC")),
            }
        )
        payload = self._call("GET", f"/api/v1/projects/{project_id}/statistics?{query}")
        metrics = decode_model(ReportMetrics, payload["metrics"])
        series = {
            str(key): decode_model(ReportSeries, value)
            for key, value in dict(payload.get("series", {})).items()
        }
        return metrics, series

    def _stream_revision(self, project_id: object, stream_id: str) -> int:
        return max(
            (
                event.revision
                for event in self.history(project_id)
                if event.stream_id == stream_id
            ),
            default=0,
        )
