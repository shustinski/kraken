"""Application PluginJobGateway backed by the authenticated local Agent API."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from kraken_agent.jobs import StagingWorkspace
from kraken_core.plugin_protocol import (
    PLUGIN_PROTOCOL_VERSION,
    PluginFrameInput,
    PluginJobManifest,
)
from kraken_manager.domain.common import ArtifactVersionId, FrameId, PluginJobId
from kraken_manager.domain.workflows import PluginJobManifestV1


class AgentUnavailable(RuntimeError):
    pass


class AgentPluginGateway:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        staging_root: Path | str,
        source_for_version: Callable[[ArtifactVersionId], Path],
        coordinate_for_frame: Callable[[FrameId], tuple[int, int]],
        media_type_for_version: Callable[[ArtifactVersionId], str],
        capabilities: frozenset[str],
        timeout: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.staging_root = Path(staging_root)
        self.source_for_version = source_for_version
        self.coordinate_for_frame = coordinate_for_frame
        self.media_type_for_version = media_type_for_version
        self.capabilities = capabilities
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
                result = json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise AgentUnavailable("Kraken Agent is unavailable or rejected the request") from exc
        if not isinstance(result, dict):
            raise AgentUnavailable("Kraken Agent returned an invalid response")
        return result

    def submit(self, manifest: PluginJobManifestV1) -> None:
        if not self.is_available(manifest.capability, manifest.protocol_version):
            raise AgentUnavailable(f"Agent has no compatible capability {manifest.capability}")
        workspace = StagingWorkspace(self.staging_root, str(manifest.job_id))
        workspace.create()
        inputs: list[PluginFrameInput] = []
        for item in manifest.inputs:
            x, y = self.coordinate_for_frame(item.frame_id)
            inputs.append(
                PluginFrameInput(
                    frame_id=str(item.frame_id),
                    x=x,
                    y=y,
                    artifact_version_id=str(item.artifact_version_id),
                    sha256=item.sha256,
                    media_type=self.media_type_for_version(item.artifact_version_id),
                    relative_path=item.relative_path,
                )
            )
            workspace.stage_file(
                self.source_for_version(item.artifact_version_id),
                item.relative_path,
                expected_sha256=item.sha256,
            )
        transport = PluginJobManifest(
            job_id=str(manifest.job_id),
            operation=manifest.capability,
            project_id=str(manifest.project_id),
            layer_id=str(manifest.layer_id),
            actor_id=str(manifest.actor_principal_id),
            target_representation_id=str(manifest.target_representation_id),
            inputs=tuple(inputs),
            parameters=dict(manifest.parameters),
            protocol_version=manifest.protocol_version,
        )
        workspace.write_manifest(transport)
        self._request("POST", "/api/v1/jobs", transport.to_dict())

    def cancel(self, job_id: PluginJobId) -> None:
        current = self._request("GET", f"/api/v1/jobs/{job_id}")
        self._request(
            "POST",
            f"/api/v1/jobs/{job_id}/cancel",
            {"expected_revision": int(current["revision"])},
        )

    def is_available(self, capability: str, protocol_version: str) -> bool:
        if protocol_version != PLUGIN_PROTOCOL_VERSION or capability not in self.capabilities:
            return False
        try:
            health = self._request("GET", "/api/v1/health")
        except AgentUnavailable:
            return False
        return health.get("status") == "ok" and health.get("api_version") == "v1"


__all__ = ["AgentPluginGateway", "AgentUnavailable"]

