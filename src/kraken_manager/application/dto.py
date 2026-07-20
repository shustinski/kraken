"""Framework-neutral command and result data transfer objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from kraken_manager.domain.artifacts import ArtifactScope, ArtifactVersion, BlobRef
from kraken_manager.domain.common import (
    ArtifactSeriesId,
    ArtifactVersionId,
    FrameId,
    LayerId,
    PerformerId,
    PluginJobId,
    PrincipalId,
    ProjectId,
    RepresentationId,
    ReviewBatchId,
    as_utc,
    new_uuid,
    require_non_empty,
    validate_uuid,
)
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.identity import ProjectRole
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind
from kraken_manager.domain.selection import FrameSelectionV1
from kraken_manager.domain.workflows import PluginInputV1, PluginJob, PluginResultManifestV1
from kraken_manager.domain.workflows import (
    ReviewBatch,
    ReviewComparisonReport,
    ReviewItem,
    ReviewPackageManifestV1,
)


class StorageBackendKind(StrEnum):
    FILESYSTEM = "filesystem"
    POSTGRESQL = "postgresql"
    CUSTOM = "custom"


class StorageScope(StrEnum):
    LOCAL = "local"
    SHARED = "shared"


@dataclass(frozen=True, slots=True)
class CommandContext:
    actor: Principal
    idempotency_key: str
    correlation_id: str = field(default_factory=new_uuid)
    performer_id: PerformerId | None = None
    effective_at: datetime | None = None
    gitlab_identity_verified: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.actor, Principal):
            raise ValueError("command actor must be an authenticated Principal")
        object.__setattr__(
            self,
            "idempotency_key",
            require_non_empty(self.idempotency_key, field="idempotency_key", maximum=255),
        )
        object.__setattr__(self, "correlation_id", validate_uuid(self.correlation_id, field="correlation_id"))
        if self.performer_id is not None:
            object.__setattr__(
                self,
                "performer_id",
                PerformerId(validate_uuid(str(self.performer_id), field="performer_id")),
            )
        if self.effective_at is not None:
            object.__setattr__(self, "effective_at", as_utc(self.effective_at, field="effective_at"))
        if not isinstance(self.gitlab_identity_verified, bool):
            raise ValueError("gitlab_identity_verified must be boolean")


@dataclass(frozen=True, slots=True)
class CreateProjectCommand:
    context: CommandContext
    name: str
    width: int
    height: int
    orientation: GridOrientation
    storage_profile_id: str
    # A client-generated stable ID makes create-project retries idempotent even
    # before any project catalog row exists.
    project_id: ProjectId = field(default_factory=lambda: ProjectId(new_uuid()))


@dataclass(frozen=True, slots=True)
class CreateLayerCommand:
    context: CommandContext
    project_id: ProjectId
    name: str
    type: LayerType
    order: int
    expected_project_revision: int


@dataclass(frozen=True, slots=True)
class CreateRepresentationCommand:
    context: CommandContext
    project_id: ProjectId
    layer_id: LayerId
    name: str
    kind: RepresentationKind
    expected_layer_revision: int
    note: str = ""
    source: str | None = None
    active: bool = False


@dataclass(frozen=True, slots=True)
class RenameProjectCommand:
    context: CommandContext
    project_id: ProjectId
    name: str
    expected_revision: int


@dataclass(frozen=True, slots=True)
class ArchiveProjectCommand:
    context: CommandContext
    project_id: ProjectId
    expected_revision: int


@dataclass(frozen=True, slots=True)
class RestoreProjectCommand:
    context: CommandContext
    project_id: ProjectId
    expected_revision: int


@dataclass(frozen=True, slots=True)
class RenameLayerCommand:
    context: CommandContext
    project_id: ProjectId
    layer_id: LayerId
    name: str
    expected_revision: int


@dataclass(frozen=True, slots=True)
class ReorderLayerCommand:
    context: CommandContext
    project_id: ProjectId
    layer_id: LayerId
    order: int
    expected_revision: int


@dataclass(frozen=True, slots=True)
class ArchiveLayerCommand:
    context: CommandContext
    project_id: ProjectId
    layer_id: LayerId
    expected_revision: int


@dataclass(frozen=True, slots=True)
class AssignProjectRoleCommand:
    context: CommandContext
    project_id: ProjectId
    principal_id: PrincipalId
    role: ProjectRole
    expected_revision: int


@dataclass(frozen=True, slots=True)
class RevokeProjectRoleCommand:
    context: CommandContext
    project_id: ProjectId
    principal_id: PrincipalId
    role: ProjectRole
    expected_revision: int


@dataclass(frozen=True, slots=True)
class RenameRepresentationCommand:
    context: CommandContext
    project_id: ProjectId
    layer_id: LayerId
    representation_id: RepresentationId
    name: str
    expected_layer_revision: int
    expected_representation_revision: int


@dataclass(frozen=True, slots=True)
class UpdateRepresentationNoteCommand:
    context: CommandContext
    project_id: ProjectId
    layer_id: LayerId
    representation_id: RepresentationId
    note: str
    expected_layer_revision: int
    expected_representation_revision: int


@dataclass(frozen=True, slots=True)
class ActivateRepresentationCommand:
    context: CommandContext
    project_id: ProjectId
    layer_id: LayerId
    representation_id: RepresentationId
    expected_layer_revision: int
    expected_representation_revision: int


@dataclass(frozen=True, slots=True)
class ArchiveRepresentationCommand:
    context: CommandContext
    project_id: ProjectId
    layer_id: LayerId
    representation_id: RepresentationId
    expected_layer_revision: int
    expected_representation_revision: int


@dataclass(frozen=True, slots=True)
class AddArtifactVersionCommand:
    context: CommandContext
    project_id: ProjectId
    series_id: ArtifactSeriesId
    filename: str
    media_type: str
    expected_series_revision: int
    parent_version_id: ArtifactVersionId | None = None
    input_version_ids: tuple[ArtifactVersionId, ...] = ()
    tool_name: str | None = None
    tool_version: str | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)
    expected_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class CreateArtifactSeriesCommand:
    context: CommandContext
    project_id: ProjectId
    scope: ArtifactScope
    name: str
    layer_id: LayerId | None = None
    representation_id: RepresentationId | None = None
    frame_id: FrameId | None = None
    series_id: ArtifactSeriesId = field(default_factory=lambda: ArtifactSeriesId(new_uuid()))


@dataclass(frozen=True, slots=True)
class SubmitPluginJobCommand:
    context: CommandContext
    project_id: ProjectId
    layer_id: LayerId
    target_representation_id: RepresentationId
    selection: FrameSelectionV1
    capability: str
    inputs: tuple[PluginInputV1, ...]
    parameters: Mapping[str, object] = field(default_factory=dict)
    protocol_version: str = "1.0"
    job_id: PluginJobId = field(default_factory=lambda: PluginJobId(new_uuid()))

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))


@dataclass(frozen=True, slots=True)
class ImportPluginResultCommand:
    context: CommandContext
    manifest: PluginResultManifestV1
    confirm_partial: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, PluginResultManifestV1):
            raise ValueError("manifest must be PluginResultManifestV1")
        if not isinstance(self.confirm_partial, bool):
            raise ValueError("confirm_partial must be boolean")


@dataclass(frozen=True, slots=True)
class PluginResultImport:
    job: PluginJob
    versions: tuple[ArtifactVersion, ...] = ()
    requires_partial_confirmation: bool = False
    awaiting_authorization: bool = False
    already_imported: bool = False


@dataclass(frozen=True, slots=True)
class CreateReviewBatchCommand:
    context: CommandContext
    project_id: ProjectId
    layer_id: LayerId
    selection: FrameSelectionV1
    items: tuple[ReviewItem, ...]
    assignee_id: PerformerId
    expected_layer_revision: int
    instructions: str = ""
    due_at: datetime | None = None
    batch_id: ReviewBatchId = field(default_factory=lambda: ReviewBatchId(new_uuid()))


@dataclass(frozen=True, slots=True)
class PlanReviewPackageCommand:
    context: CommandContext
    project_id: ProjectId
    batch_id: ReviewBatchId
    expected_batch_revision: int
    include_images: bool = True
    package_id: ReviewBatchId = field(default_factory=lambda: ReviewBatchId(new_uuid()))


@dataclass(frozen=True, slots=True)
class ExportReviewPackageCommand:
    context: CommandContext
    project_id: ProjectId
    batch_id: ReviewBatchId
    expected_batch_revision: int
    destination: str
    include_images: bool = True
    package_id: ReviewBatchId = field(default_factory=lambda: ReviewBatchId(new_uuid()))


@dataclass(frozen=True, slots=True)
class DryRunReviewReturnCommand:
    context: CommandContext
    project_id: ProjectId
    batch_id: ReviewBatchId
    expected_batch_revision: int
    source: str


@dataclass(frozen=True, slots=True)
class CommitReviewReturnCommand:
    context: CommandContext
    project_id: ProjectId
    batch_id: ReviewBatchId
    expected_batch_revision: int
    source: str


@dataclass(frozen=True, slots=True)
class AcceptReviewCommand:
    context: CommandContext
    project_id: ProjectId
    batch_id: ReviewBatchId
    expected_batch_revision: int
    candidate_version_ids: tuple[ArtifactVersionId, ...]


@dataclass(frozen=True, slots=True)
class RequestReviewChangesCommand:
    context: CommandContext
    project_id: ProjectId
    batch_id: ReviewBatchId
    expected_batch_revision: int
    reason: str


@dataclass(frozen=True, slots=True)
class ReviewPackagePlan:
    manifest: ReviewPackageManifestV1
    total_size_bytes: int
    issues: tuple[str, ...] = ()

    @property
    def can_export(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class ReviewReturnPlan:
    batch_id: ReviewBatchId
    package_id: ReviewBatchId
    report: ReviewComparisonReport
    return_fingerprint: str


@dataclass(frozen=True, slots=True)
class ReviewReturnCommitResult:
    batch: ReviewBatch
    report: ReviewComparisonReport
    candidate_versions: tuple[ArtifactVersion, ...]


@dataclass(frozen=True, slots=True)
class ReturnedFileDigest:
    relative_path: str
    sha256: str | None
    size_bytes: int | None = None
    valid: bool = True

    def __post_init__(self) -> None:
        if self.size_bytes is not None and (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("returned file size must be non-negative or None")
        if not isinstance(self.valid, bool):
            raise ValueError("returned file valid flag must be boolean")


@dataclass(frozen=True, slots=True)
class ActiveVersionSnapshot:
    frame_id: str
    version_id: ArtifactVersionId


@dataclass(frozen=True, slots=True)
class ProblemDetails:
    """Stable application error shape suitable for RFC 9457 API adapters."""

    code: str
    title: str
    detail: str
    status: int
    fields: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 400 <= self.status <= 599:
            raise ValueError("problem status must be an HTTP error status")


@dataclass(frozen=True, slots=True)
class StoredContent:
    """Result of hashing/storing an immutable stream."""

    blob: BlobRef
    already_existed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.blob, BlobRef):
            raise ValueError("stored content requires a BlobRef")
        if not isinstance(self.already_existed, bool):
            raise ValueError("already_existed must be boolean")


__all__ = [
    "AcceptReviewCommand",
    "ActivateRepresentationCommand",
    "ActiveVersionSnapshot",
    "AddArtifactVersionCommand",
    "ArchiveLayerCommand",
    "ArchiveProjectCommand",
    "ArchiveRepresentationCommand",
    "AssignProjectRoleCommand",
    "CommitReviewReturnCommand",
    "CommandContext",
    "CreateLayerCommand",
    "CreateProjectCommand",
    "CreateRepresentationCommand",
    "CreateArtifactSeriesCommand",
    "ImportPluginResultCommand",
    "PluginResultImport",
    "CreateReviewBatchCommand",
    "DryRunReviewReturnCommand",
    "ExportReviewPackageCommand",
    "PlanReviewPackageCommand",
    "ProblemDetails",
    "RequestReviewChangesCommand",
    "RenameLayerCommand",
    "RenameProjectCommand",
    "RenameRepresentationCommand",
    "ReorderLayerCommand",
    "RestoreProjectCommand",
    "RevokeProjectRoleCommand",
    "ReturnedFileDigest",
    "ReviewPackagePlan",
    "ReviewReturnCommitResult",
    "ReviewReturnPlan",
    "StorageBackendKind",
    "StorageScope",
    "StoredContent",
    "SubmitPluginJobCommand",
    "UpdateRepresentationNoteCommand",
]
