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
    PluginAsset,
    PluginJobManifest,
    PluginJobManifestV2,
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
        v2_capabilities: frozenset[str] = frozenset(),
        timeout: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.staging_root = Path(staging_root)
        self.source_for_version = source_for_version
        self.coordinate_for_frame = coordinate_for_frame
        self.media_type_for_version = media_type_for_version
        self.capabilities = capabilities
        self.v2_capabilities = v2_capabilities
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
        v2_inputs: list[PluginAsset] = []
        for item in manifest.inputs:
            x, y = self.coordinate_for_frame(item.frame_id)
            media_type = self.media_type_for_version(item.artifact_version_id)
            inputs.append(
                PluginFrameInput(
                    frame_id=str(item.frame_id),
                    x=x,
                    y=y,
                    artifact_version_id=str(item.artifact_version_id),
                    sha256=item.sha256,
                    media_type=media_type,
                    relative_path=item.relative_path,
                )
            )
            v2_inputs.append(
                PluginAsset(
                    asset_id=str(item.artifact_version_id),
                    role=(
                        "image"
                        if media_type.startswith("image/")
                        else "model"
                        if media_type in {"application/x-pytorch", "application/onnx"}
                        else "input"
                    ),
                    scope="frame",
                    relative_path=item.relative_path,
                    sha256=item.sha256,
                    media_type=media_type,
                    frame_id=str(item.frame_id),
                    representation_id=str(manifest.target_representation_id),
                    x=x,
                    y=y,
                )
            )
            workspace.stage_file(
                self.source_for_version(item.artifact_version_id),
                item.relative_path,
                expected_sha256=item.sha256,
            )
        transport = (
            PluginJobManifestV2(
                job_id=str(manifest.job_id),
                operation=manifest.capability,
                project_id=str(manifest.project_id),
                layer_id=str(manifest.layer_id),
                actor_id=str(manifest.actor_principal_id),
                inputs=tuple(v2_inputs),
                parameters=dict(manifest.parameters),
                target_representation_ids=(str(manifest.target_representation_id),),
            )
            if manifest.capability in self.v2_capabilities
            else PluginJobManifest(
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

    def get_job(self, job_id: PluginJobId | str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/jobs/{job_id}")

    def get_result(self, job_id: PluginJobId | str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/jobs/{job_id}/result")

    def get_publications(self, job_id: PluginJobId | str) -> tuple[dict[str, Any], ...]:
        payload = self._request("GET", f"/api/v1/jobs/{job_id}/publications")
        publications = payload.get("publications", ())
        if not isinstance(publications, list):
            raise AgentUnavailable("Kraken Agent returned invalid publications")
        return tuple(
            dict(item)
            for item in publications
            if isinstance(item, dict)
        )

    def shutdown(self) -> None:
        self._request("POST", "/api/v1/shutdown", {})

    def complete_import(self, job_id: PluginJobId | str) -> dict[str, Any]:
        current = self.get_job(job_id)
        return self._request(
            "POST",
            f"/api/v1/jobs/{job_id}/complete-import",
            {"expected_revision": int(current["revision"])},
        )

    def confirm_partial(self, job_id: PluginJobId | str) -> dict[str, Any]:
        current = self.get_job(job_id)
        return self._request(
            "POST",
            f"/api/v1/jobs/{job_id}/confirm-partial",
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

