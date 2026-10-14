"""plugin_jobs: manifest payload."""
from __future__ import annotations
from collections.abc import Mapping
from kraken_manager.application.errors import ConflictError
from kraken_manager.domain.common import ArtifactVersionId, FrameId, PluginJobId
from kraken_manager.domain.selection import FrameSelectionV1
from kraken_manager.domain.workflows import (
    PluginInputV1,
    PluginJobManifestV1,
)

from kraken_manager.application.plugin_jobs_parts.plain import _plain

def _manifest_payload(manifest: PluginJobManifestV1) -> dict[str, object]:
    return {
        "schema_version": manifest.SCHEMA_VERSION,
        "protocol_version": manifest.protocol_version,
        "job_id": str(manifest.job_id),
        "project_id": str(manifest.project_id),
        "layer_id": str(manifest.layer_id),
        "target_representation_id": str(manifest.target_representation_id),
        "selection": manifest.selection.to_dict(),
        "actor_principal_id": str(manifest.actor_principal_id),
        "capability": manifest.capability,
        "inputs": [
            {
                "frame_id": str(item.frame_id),
                "artifact_version_id": str(item.artifact_version_id),
                "sha256": item.sha256,
                "relative_path": item.relative_path,
            }
            for item in manifest.inputs
        ],
        "parameters": _plain(manifest.parameters),
    }

def _manifest_from_payload(payload: Mapping[str, object]) -> PluginJobManifestV1:
    raw_inputs = payload.get("inputs")
    raw_selection = payload.get("selection")
    if not isinstance(raw_inputs, (tuple, list)) or not isinstance(raw_selection, Mapping):
        raise ConflictError("Stored plugin job manifest is damaged")
    try:
        inputs = tuple(
            PluginInputV1(
                frame_id=FrameId(str(item["frame_id"])),
                artifact_version_id=ArtifactVersionId(str(item["artifact_version_id"])),
                sha256=str(item["sha256"]),
                relative_path=str(item["relative_path"]),
            )
            for item in raw_inputs
            if isinstance(item, Mapping)
        )
        if len(inputs) != len(raw_inputs):
            raise ValueError("invalid input row")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("invalid parameters")
        return PluginJobManifestV1(
            job_id=PluginJobId(str(payload["job_id"])),
            project_id=str(payload["project_id"]),  # type: ignore[arg-type]
            layer_id=str(payload["layer_id"]),  # type: ignore[arg-type]
            target_representation_id=str(payload["target_representation_id"]),  # type: ignore[arg-type]
            selection=FrameSelectionV1.from_dict(raw_selection),
            actor_principal_id=str(payload["actor_principal_id"]),  # type: ignore[arg-type]
            capability=str(payload["capability"]),
            inputs=inputs,
            parameters=dict(parameters),
            protocol_version=str(payload.get("protocol_version", "1.0")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConflictError("Stored plugin job manifest is damaged") from exc
