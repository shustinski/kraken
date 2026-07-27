"""Semantic application ports implemented by filesystem, SQL, and service adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import ContextManager, Protocol, Self, runtime_checkable

from kraken_manager.application.dto import StorageBackendKind, StorageScope, StoredContent
from kraken_manager.domain.artifacts import ArtifactSeries, ArtifactVersion, BlobRef
from kraken_manager.domain.common import (
    ArtifactSeriesId,
    ArtifactVersionId,
    LayerId,
    PerformerId,
    PluginJobId,
    PrincipalId,
    ProjectId,
    RepresentationId,
    ReviewBatchId,
)
from kraken_manager.domain.events import EventEnvelope
from kraken_manager.domain.identity import Performer, Principal, ProjectRole, ProjectRoleAssignment
from kraken_manager.domain.project import Layer, Project, Representation
from kraken_manager.domain.workflows import (
    PluginJob,
    PluginJobManifestV1,
    PluginResultManifestV1,
    ReviewBatch,
    ReviewPackageManifestV1,
)


@dataclass(frozen=True, slots=True)
class StorageCapabilities:
    multi_writer: bool
    transactions: bool
    snapshots: bool
    streaming: bool
    external_references: bool
    max_frames: int | None

    def __post_init__(self) -> None:
        for field_name in (
            "multi_writer",
            "transactions",
            "snapshots",
            "streaming",
            "external_references",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        if self.max_frames is not None and (
            isinstance(self.max_frames, bool)
            or not isinstance(self.max_frames, int)
            or self.max_frames < 1
        ):
            raise ValueError("max_frames must be positive or None")


@dataclass(frozen=True, slots=True)
class StorageProfile:
    id: str
    name: str
    metadata_backend: StorageBackendKind
    blob_backend: str
    scope: StorageScope
    capabilities: StorageCapabilities

    def __post_init__(self) -> None:
        profile_id = self.id.strip()
        name = self.name.strip()
        blob_backend = self.blob_backend.strip()
        if not profile_id or not name or not blob_backend:
            raise ValueError("storage profile id, name, and blob backend must not be empty")
        object.__setattr__(self, "id", profile_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "blob_backend", blob_backend)
        if not isinstance(self.metadata_backend, StorageBackendKind):
            object.__setattr__(self, "metadata_backend", StorageBackendKind(self.metadata_backend))
        if not isinstance(self.scope, StorageScope):
            object.__setattr__(self, "scope", StorageScope(self.scope))
        if not isinstance(self.capabilities, StorageCapabilities):
            raise ValueError("storage profile capabilities must be a StorageCapabilities value")

    @property
    def shared(self) -> bool:
        return self.scope is StorageScope.SHARED


@runtime_checkable
class EventStore(Protocol):
    def load_stream(
        self,
        stream_id: str,
        *,
        after_revision: int = 0,
        as_of: datetime | None = None,
    ) -> tuple[EventEnvelope, ...]: ...

    def current_revision(self, stream_id: str) -> int: ...

    def append(
        self,
        stream_id: str,
        *,
        expected_revision: int,
        events: Sequence[EventEnvelope],
    ) -> int: ...

    def find_by_idempotency_key(
        self,
        project_id: ProjectId,
        idempotency_key: str,
    ) -> tuple[EventEnvelope, ...]: ...


@runtime_checkable
class ProjectionStore(Protocol):
    def get_project(self, project_id: ProjectId, *, as_of: datetime | None = None) -> Project | None: ...

    def save_project(self, project: Project) -> None: ...

    def get_layer(self, layer_id: LayerId, *, as_of: datetime | None = None) -> Layer | None: ...

    def list_layers(
        self,
        project_id: ProjectId,
        *,
        include_archived: bool = False,
        as_of: datetime | None = None,
    ) -> tuple[Layer, ...]: ...

    def save_layer(self, layer: Layer) -> None: ...

    def get_representation(
        self,
        representation_id: RepresentationId,
        *,
        as_of: datetime | None = None,
    ) -> Representation | None: ...

    def save_representation(self, representation: Representation) -> None: ...

    def list_representations(
        self,
        layer_id: LayerId,
        *,
        include_archived: bool = False,
        as_of: datetime | None = None,
    ) -> tuple[Representation, ...]: ...

    def get_artifact_series(
        self, series_id: ArtifactSeriesId, *, as_of: datetime | None = None
    ) -> ArtifactSeries | None: ...

    def list_artifact_series(
        self,
        project_id: ProjectId,
        *,
        layer_id: LayerId | None = None,
        representation_id: RepresentationId | None = None,
        include_archived: bool = False,
        as_of: datetime | None = None,
    ) -> tuple[ArtifactSeries, ...]: ...

    def save_artifact_series(self, series: ArtifactSeries) -> None: ...

    def get_artifact_version(
        self, version_id: ArtifactVersionId, *, as_of: datetime | None = None
    ) -> ArtifactVersion | None: ...

    def get_active_artifact_version(
        self, series_id: ArtifactSeriesId, *, as_of: datetime | None = None
    ) -> ArtifactVersion | None: ...

    def save_artifact_version(self, version: ArtifactVersion, *, activate: bool) -> None: ...

    def save_plugin_job(self, job: PluginJob) -> None: ...

    def get_plugin_job(self, job_id: PluginJobId, *, as_of: datetime | None = None) -> PluginJob | None: ...

    def get_review_batch(
        self, batch_id: ReviewBatchId, *, as_of: datetime | None = None
    ) -> ReviewBatch | None: ...

    def save_review_batch(self, batch: ReviewBatch) -> None: ...

    def list_active_review_batches(
        self,
        project_id: ProjectId,
        layer_id: LayerId,
        *,
        as_of: datetime | None = None,
    ) -> tuple[ReviewBatch, ...]: ...


@runtime_checkable
class BlobStore(Protocol):
    def put(
        self,
        chunks: Iterable[bytes],
        *,
        expected_sha256: str | None = None,
    ) -> StoredContent: ...

    def iter_bytes(self, reference: BlobRef, *, chunk_size: int = 1024 * 1024) -> Iterator[bytes]: ...

    def exists(self, reference: BlobRef) -> bool: ...


@runtime_checkable
class IdentityStore(Protocol):
    def get(self, principal_id: PrincipalId) -> Principal | None: ...

    def get_by_external_key(self, external_key: str) -> Principal | None: ...

    def save(self, principal: Principal) -> None: ...


@runtime_checkable
class PerformerStore(Protocol):
    """Catalog of attribution targets, independent from authentication.

    A performer may be manual (``principal_id is None``).  At most one
    performer may be linked to a given principal.  ``create`` rejects an
    existing id or principal link, while ``update`` rejects an unknown id.
    ``list`` is ordered by Unicode-casefolded name and then performer id so
    adapters expose the same deterministic catalog order.
    """

    def get(self, performer_id: PerformerId) -> Performer | None: ...

    def get_by_principal(self, principal_id: PrincipalId) -> Performer | None: ...

    def list(self, *, include_archived: bool = False) -> tuple[Performer, ...]: ...

    def create(self, performer: Performer) -> Performer: ...

    def update(self, performer: Performer) -> Performer: ...

    def archive(self, performer_id: PerformerId) -> Performer: ...


@runtime_checkable
class AclStore(Protocol):
    def roles_for(self, project_id: ProjectId, principal_id: PrincipalId) -> frozenset[ProjectRole]: ...

    def assign(self, assignment: ProjectRoleAssignment) -> None: ...

    def revoke(self, project_id: ProjectId, principal_id: PrincipalId, role: ProjectRole) -> None: ...


@runtime_checkable
class UnitOfWork(Protocol):
    event_store: EventStore
    projections: ProjectionStore
    blobs: BlobStore
    identities: IdentityStore
    acl: AclStore

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool | None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class UnitOfWorkFactory(Protocol):
    def __call__(self) -> UnitOfWork: ...


class StorageProfileCatalog(Protocol):
    def get(self, profile_id: str) -> StorageProfile | None: ...

    def list(self) -> tuple[StorageProfile, ...]: ...


class LockManager(Protocol):
    def acquire(self, key: str, *, timeout_seconds: float | None = None) -> ContextManager[None]: ...


class BackgroundJobQueue(Protocol):
    def enqueue(self, job: PluginJob) -> None: ...

    def get(self, job_id: PluginJobId) -> PluginJob | None: ...

    def lease_next(self, *, worker_id: str, lease_until: datetime) -> PluginJob | None: ...

    def acknowledge(self, job_id: PluginJobId, *, worker_id: str) -> None: ...


class ReviewPackageWriter(Protocol):
    def write(
        self,
        destination: str,
        manifest: ReviewPackageManifestV1,
        files: Mapping[str, Callable[[], Iterator[bytes]]],
    ) -> None: ...


class ReviewPackageReader(Protocol):
    def read_manifest(self, source: str) -> ReviewPackageManifestV1: ...

    def iter_file(self, source: str, relative_path: str) -> Iterator[bytes]: ...

    def list_relative_paths(self, source: str) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class SnapshotChunk:
    kind: str
    key: str
    sequence: int
    payload: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class SnapshotCheckpoint:
    project_id: ProjectId
    last_sequence: int
    token: str
    complete: bool


class StorageSnapshotExporter(Protocol):
    def export_snapshot(
        self,
        project_id: ProjectId,
        *,
        resume_from: SnapshotCheckpoint | None = None,
    ) -> Iterator[SnapshotChunk]: ...


class StorageSnapshotImporter(Protocol):
    def import_snapshot(
        self,
        chunks: Iterable[SnapshotChunk],
        *,
        resume_from: SnapshotCheckpoint | None = None,
    ) -> SnapshotCheckpoint: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class Hasher(Protocol):
    def sha256(self, chunks: Iterable[bytes]) -> BlobRef: ...


class SecretStore(Protocol):
    def get(self, key: str) -> bytes | None: ...

    def set(self, key: str, value: bytes) -> None: ...

    def delete(self, key: str) -> None: ...


class PluginJobGateway(Protocol):
    def submit(self, manifest: PluginJobManifestV1) -> None: ...

    def cancel(self, job_id: PluginJobId) -> None: ...

    def is_available(self, capability: str, protocol_version: str) -> bool: ...


class PluginResultContentReader(Protocol):
    """Read immutable output bytes from an Agent-owned staging workspace."""

    def iter_output(
        self,
        manifest: PluginResultManifestV1,
        relative_path: str,
    ) -> Iterator[bytes]: ...


__all__ = [
    "AclStore",
    "BackgroundJobQueue",
    "BlobStore",
    "Clock",
    "EventStore",
    "Hasher",
    "IdentityStore",
    "LockManager",
    "PerformerStore",
    "PluginJobGateway",
    "PluginResultContentReader",
    "ProjectionStore",
    "ReviewPackageReader",
    "ReviewPackageWriter",
    "SecretStore",
    "SnapshotCheckpoint",
    "SnapshotChunk",
    "StorageCapabilities",
    "StorageProfile",
    "StorageProfileCatalog",
    "StorageSnapshotExporter",
    "StorageSnapshotImporter",
    "UnitOfWork",
    "UnitOfWorkFactory",
]
