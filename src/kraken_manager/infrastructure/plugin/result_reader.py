"""Translate Agent V1 results and stream their staging-only output bytes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from kraken_agent.jobs import StagingWorkspace
from kraken_core.plugin_protocol import PluginJobOutcome, PluginResultManifest, safe_relative_path
from kraken_core.safe_files import open_regular_read
from kraken_manager.domain.common import FrameId, PluginJobId
from kraken_manager.domain.workflows import (
    PluginFrameOutcome,
    PluginFrameResultV1,
    PluginResultManifestV1,
    PluginResultOutcome,
)


def domain_result_from_transport(transport: PluginResultManifest) -> PluginResultManifestV1:
    """Normalize plugin-controlled identifiers at the trust boundary."""
    outcome = PluginResultOutcome(PluginJobOutcome(transport.outcome).value)
    results = tuple(
        PluginFrameResultV1(
            output_id=str(
                uuid5(
                    NAMESPACE_URL,
                    f"kraken:plugin-output:{transport.job_id}:{output.output_id}",
                )
            ),
            frame_id=FrameId(output.frame_id),
            outcome=PluginFrameOutcome.SUCCEEDED,
            relative_path=output.relative_path,
            sha256=output.sha256,
            media_type=output.media_type,
            role=output.role,
            warning="; ".join(output.warnings) or None,
        )
        for output in transport.outputs
    )
    parameters = dict(transport.applied_parameters)
    if transport.errors:
        parameters["transport_errors"] = list(transport.errors)
    return PluginResultManifestV1(
        job_id=PluginJobId(transport.job_id),
        plugin_name=transport.plugin_id,
        plugin_version=transport.plugin_version,
        results=results,
        parameters_applied=parameters,
        outcome=outcome,
        protocol_version=transport.protocol_version,
    )


class AgentStagingResultContentReader:
    """PluginResultContentReader over one Agent staging root."""

    def __init__(self, staging_root: Path | str, *, chunk_size: int = 1024 * 1024) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.staging_root = Path(staging_root)
        self.chunk_size = chunk_size

    def iter_output(self, manifest: PluginResultManifestV1, relative_path: str) -> Iterator[bytes]:
        normalized = safe_relative_path(relative_path)
        if not normalized.startswith("outputs/"):
            raise ValueError("Plugin result content must be below outputs/")
        workspace = StagingWorkspace(self.staging_root, str(manifest.job_id))
        candidate = workspace.resolve(normalized)
        with open_regular_read(candidate, root=workspace.path) as stream:
            while chunk := stream.read(self.chunk_size):
                yield chunk


__all__ = ["AgentStagingResultContentReader", "domain_result_from_transport"]
