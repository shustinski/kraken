"""persistent_services: layer source scan."""
from __future__ import annotations
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from typing import Any
from kraken_manager.application.dto import (
    StorageBackendKind,
    StorageScope,
)
from kraken_manager.application.ports import StorageCapabilities, StorageProfile
from kraken_manager.domain.common import (
    LayerId,
    ProjectId,
    validate_uuid,
)
from kraken_manager.domain.project import (
    Layer,
    Project,
    Representation,
)
from kraken_core.plugin_protocol import safe_relative_path

class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

class _AgentBlobResultReader:
    def __init__(self, blobs: Any, references: Mapping[str, Any]) -> None:
        self.blobs = blobs
        self.references = dict(references)

    def iter_output(self, manifest: Any, relative_path: str) -> Iterator[bytes]:
        del manifest
        normalized = safe_relative_path(relative_path)
        reference = self.references.get(normalized)
        if reference is None:
            raise FileNotFoundError(f"Agent output was not uploaded: {normalized}")
        return self.blobs.iter_bytes(reference)

class ServerStorageProfiles:
    """Server-owned profile catalog; persistence/configuration may replace it."""

    def __init__(self, *, profile_id: str = "server-postgres", max_frames: int | None = None) -> None:
        self.profile = StorageProfile(
            id=profile_id,
            name="Kraken Server PostgreSQL",
            metadata_backend=StorageBackendKind.POSTGRESQL,
            blob_backend="filesystem",
            scope=StorageScope.SHARED,
            capabilities=StorageCapabilities(
                multi_writer=True,
                transactions=True,
                snapshots=True,
                streaming=True,
                external_references=True,
                max_frames=max_frames,
            ),
        )

    def get(self, profile_id: str) -> StorageProfile | None:
        return self.profile if profile_id == self.profile.id else None

    def list(self) -> tuple[StorageProfile, ...]:
        return (self.profile,)

def _project_dict(project: Project) -> dict[str, Any]:
    return {
        "project_id": str(project.id),
        "name": project.name,
        "width": project.width,
        "height": project.height,
        "orientation": project.orientation.value,
        "storage_profile": project.storage_profile,
        "state": project.state.value,
        "revision": project.revision,
        "created_at": project.created_at.isoformat(),
    }

def _layer_dict(layer: Layer) -> dict[str, Any]:
    return {
        "layer_id": str(layer.id),
        "project_id": str(layer.project_id),
        "name": layer.name,
        "type": layer.type.value,
        "order": layer.order,
        "state": layer.state.value,
        "revision": layer.revision,
        "created_at": layer.created_at.isoformat(),
    }

def _representation_dict(representation: Representation) -> dict[str, Any]:
    return {
        "representation_id": str(representation.id),
        "project_id": str(representation.project_id),
        "layer_id": str(representation.layer_id),
        "name": representation.name,
        "kind": representation.kind.value,
        "purpose": representation.purpose.value,
        "source_image_representation_id": (
            None
            if representation.source_image_representation_id is None
            else str(representation.source_image_representation_id)
        ),
        "note": representation.note,
        "source": representation.source,
        "active": representation.active,
        "state": representation.state.value,
        "revision": representation.revision,
        "created_at": representation.created_at.isoformat(),
    }

def _project_id(value: str) -> ProjectId:
    return ProjectId(validate_uuid(value, field="project_id"))

def _layer_source_scan(payload: Mapping[str, Any]) -> Any:
    from kraken_manager.workspace import LayerSourceScan

    fingerprints_raw = payload.get("file_fingerprints") or {}
    fingerprints: dict[str, tuple[int, int]] = {}
    if isinstance(fingerprints_raw, Mapping):
        for key, value in fingerprints_raw.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                fingerprints[str(key)] = (int(value[0]), int(value[1]))
    positions_raw = payload.get("frame_positions") or {}
    positions = (
        {str(key): int(value) for key, value in positions_raw.items()}
        if isinstance(positions_raw, Mapping)
        else {}
    )
    return LayerSourceScan(
        selected_root=str(payload.get("selected_root", "")),
        working_directory=str(payload.get("working_directory", "")),
        jpg_files=tuple(str(item) for item in payload.get("jpg_files") or ()),
        bmp_files=tuple(str(item) for item in payload.get("bmp_files") or ()),
        ssc_files=tuple(str(item) for item in payload.get("ssc_files") or ()),
        prv_files=tuple(str(item) for item in payload.get("prv_files") or ()),
        frame_positions=positions,
        total_files=int(payload.get("total_files") or 0),
        total_bytes=int(payload.get("total_bytes") or 0),
        issues=tuple(str(item) for item in payload.get("issues") or ()),
        file_fingerprints=fingerprints,
    )

def _layer_id(value: str) -> LayerId:
    return LayerId(validate_uuid(value, field="layer_id"))
