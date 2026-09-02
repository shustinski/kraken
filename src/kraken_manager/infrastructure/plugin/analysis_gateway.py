"""Authenticated local-Agent gateway for Karakal analysis partitions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from kraken_agent.jobs import StagingWorkspace
from kraken_agent.protocols import parse_result_json
from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest, AnalysisPartitionResultManifest
from kraken_manager.application.analysis_runs import AnalysisGatewayJob

from .agent_gateway import AgentUnavailable


class AgentAnalysisGateway:
    """Stages immutable inputs and controls jobs without exposing a DB connection."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        staging_root: Path | str,
        timeout: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.staging_root = Path(staging_root).resolve()
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise AgentUnavailable("Kraken Agent is unavailable or rejected the analysis request") from exc
        if not isinstance(value, dict):
            raise AgentUnavailable("Kraken Agent returned an invalid response")
        return value

    @staticmethod
    def _job(payload: Mapping[str, object]) -> AnalysisGatewayJob:
        progress = payload.get("progress")
        return AnalysisGatewayJob(
            job_id=str(payload.get("job_id", "")),
            state=str(payload.get("state", "")),
            revision=int(payload.get("revision", -1)),
            progress=progress if isinstance(progress, Mapping) else None,
            error=str(payload.get("error") or ""),
        )

    def submit(
        self,
        manifest: AnalysisPartitionJobManifest,
        source_paths: Mapping[str, Path],
    ) -> AnalysisGatewayJob:
        workspace = StagingWorkspace(self.staging_root, manifest.job_id)
        workspace.create()
        for frame in manifest.frames:
            for artifact in frame.artifacts:
                try:
                    source = source_paths[artifact.artifact_version_id]
                except KeyError as exc:
                    raise ValueError(
                        f"No source path for artifact version {artifact.artifact_version_id}"
                    ) from exc
                destination = workspace.resolve(artifact.relative_path)
                if destination.is_file() and workspace.digest(artifact.relative_path) == artifact.sha256:
                    continue
                workspace.stage_file(source, artifact.relative_path, expected_sha256=artifact.sha256)
        workspace.write_manifest(manifest)
        return self._job(self._request("POST", "/api/v1/jobs", manifest.to_payload()))

    def get(self, job_id: str) -> AnalysisGatewayJob:
        return self._job(self._request("GET", f"/api/v1/jobs/{job_id}"))

    def result(self, manifest: AnalysisPartitionJobManifest) -> tuple[AnalysisPartitionResultManifest, Path]:
        workspace = StagingWorkspace(self.staging_root, manifest.job_id)
        result_path = workspace.resolve("result.json")
        parsed = parse_result_json(result_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, AnalysisPartitionResultManifest) or parsed.bundle is None:
            raise ValueError("Agent did not publish a Karakal frame bundle")
        workspace.verify_result(parsed, manifest)
        return parsed, workspace.resolve(parsed.bundle.relative_path)

    def _transition(self, job: AnalysisGatewayJob, suffix: str) -> AnalysisGatewayJob:
        return self._job(
            self._request(
                "POST",
                f"/api/v1/jobs/{job.job_id}/{suffix}",
                {"expected_revision": job.revision},
            )
        )

    def confirm_partial(self, job: AnalysisGatewayJob) -> AnalysisGatewayJob:
        return self._transition(job, "confirm-partial")

    def complete_import(self, job: AnalysisGatewayJob) -> AnalysisGatewayJob:
        return self._transition(job, "complete-import")

    def cancel(self, job: AnalysisGatewayJob) -> AnalysisGatewayJob:
        return self._transition(job, "cancel")


__all__ = ["AgentAnalysisGateway"]
