"""Durable plugin and review workflow aggregates and public V1 manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, Final

from .artifacts import validate_sha256
from .common import (
    ArtifactVersionId,
    DomainValidationError,
    FrameId,
    InvalidStateTransition,
    LayerId,
    PerformerId,
    PluginJobId,
    PrincipalId,
    ProjectId,
    RepresentationId,
    ReviewBatchId,
    as_utc,
    freeze_mapping,
    new_uuid,
    require_non_empty,
    utc_now,
    validate_uuid,
)
from .selection import FrameSelectionV1


def validate_relative_manifest_path(value: str) -> str:
    """Accept only canonical forward-slash relative paths inside a package."""

    path = require_non_empty(value, field="relative_path", maximum=2048)
    if "\\" in path or "\x00" in path or path.startswith("/"):
        raise DomainValidationError("manifest path must be a canonical relative POSIX path")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts) or ":" in parts[0]:
        raise DomainValidationError("manifest path cannot escape the staging/package directory")
    return path


class FrameStatus(StrEnum):
    """Derived status for one ``project + layer + frame`` projection row."""

    EMPTY = "empty"
    IMAGE_READY = "image_ready"
    PROCESSING = "processing"
    VECTORIZED = "vectorized"
    IN_REVIEW = "in_review"
    RETURNED_UNCHANGED = "returned_unchanged"
    RETURNED_CHANGED = "returned_changed"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    CONFLICT = "conflict"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class FrameStatusFacts:
    """Facts consumed by the projection/UI status derivation."""

    has_image: bool = False
    has_vector: bool = False
    processing: bool = False
    in_review: bool = False
    returned_unchanged: bool = False
    returned_changed: bool = False
    approved: bool = False
    changes_requested: bool = False
    conflict: bool = False
    error: bool = False

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if not isinstance(getattr(self, field_name), bool):
                raise DomainValidationError(f"frame status fact {field_name} must be boolean")
        if self.returned_unchanged and self.returned_changed:
            raise DomainValidationError("a review return cannot be both changed and unchanged")


def derive_frame_status(facts: FrameStatusFacts) -> FrameStatus:
    """Apply the documented visual priority without storing a global frame status."""

    if facts.error:
        return FrameStatus.ERROR
    if facts.conflict:
        return FrameStatus.CONFLICT
    if facts.changes_requested:
        return FrameStatus.CHANGES_REQUESTED
    if facts.approved:
        return FrameStatus.APPROVED
    if facts.returned_changed:
        return FrameStatus.RETURNED_CHANGED
    if facts.returned_unchanged:
        return FrameStatus.RETURNED_UNCHANGED
    if facts.in_review:
        return FrameStatus.IN_REVIEW
    if facts.processing:
        return FrameStatus.PROCESSING
    if facts.has_vector:
        return FrameStatus.VECTORIZED
    if facts.has_image:
        return FrameStatus.IMAGE_READY
    return FrameStatus.EMPTY


class PluginJobState(StrEnum):
    QUEUED = "queued"
    STAGING = "staging"
    RUNNING = "running"
    WAITING_FOR_USER = "waiting_for_user"
    IMPORTING = "importing"
    PARTIAL = "partial"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_AUTHORIZATION = "awaiting_authorization"
    RECOVERY_REQUIRED = "recovery_required"


PLUGIN_JOB_TRANSITIONS: Final = {
    PluginJobState.QUEUED: frozenset(
        {PluginJobState.STAGING, PluginJobState.CANCELLED, PluginJobState.FAILED}
    ),
    PluginJobState.STAGING: frozenset(
        {
            PluginJobState.RUNNING,
            PluginJobState.CANCELLED,
            PluginJobState.FAILED,
            PluginJobState.RECOVERY_REQUIRED,
        }
    ),
    PluginJobState.RUNNING: frozenset(
        {
            PluginJobState.WAITING_FOR_USER,
            PluginJobState.IMPORTING,
            PluginJobState.PARTIAL,
            PluginJobState.CANCELLED,
            PluginJobState.FAILED,
            PluginJobState.AWAITING_AUTHORIZATION,
            PluginJobState.RECOVERY_REQUIRED,
        }
    ),
    PluginJobState.WAITING_FOR_USER: frozenset(
        {
            PluginJobState.RUNNING,
            PluginJobState.IMPORTING,
            PluginJobState.CANCELLED,
            PluginJobState.FAILED,
            PluginJobState.RECOVERY_REQUIRED,
        }
    ),
    PluginJobState.IMPORTING: frozenset(
        {
            PluginJobState.SUCCEEDED,
            PluginJobState.PARTIAL,
            PluginJobState.FAILED,
            PluginJobState.CANCELLED,
            PluginJobState.AWAITING_AUTHORIZATION,
            PluginJobState.RECOVERY_REQUIRED,
        }
    ),
    PluginJobState.PARTIAL: frozenset(
        {PluginJobState.IMPORTING, PluginJobState.CANCELLED, PluginJobState.FAILED}
    ),
    PluginJobState.AWAITING_AUTHORIZATION: frozenset(
        {PluginJobState.IMPORTING, PluginJobState.CANCELLED, PluginJobState.FAILED}
    ),
    PluginJobState.RECOVERY_REQUIRED: frozenset(
        {PluginJobState.QUEUED, PluginJobState.STAGING, PluginJobState.CANCELLED, PluginJobState.FAILED}
    ),
    PluginJobState.SUCCEEDED: frozenset(),
    PluginJobState.FAILED: frozenset(),
    PluginJobState.CANCELLED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class PluginJob:
    id: PluginJobId
    project_id: ProjectId
    layer_id: LayerId
    selection: FrameSelectionV1
    actor_principal_id: PrincipalId
    target_representation_id: RepresentationId
    capability: str
    state: PluginJobState
    revision: int
    progress: float
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", PluginJobId(validate_uuid(str(self.id), field="plugin_job.id")))
        object.__setattr__(
            self, "project_id", ProjectId(validate_uuid(str(self.project_id), field="plugin_job.project_id"))
        )
        object.__setattr__(
            self, "layer_id", LayerId(validate_uuid(str(self.layer_id), field="plugin_job.layer_id"))
        )
        object.__setattr__(
            self,
            "actor_principal_id",
            PrincipalId(validate_uuid(str(self.actor_principal_id), field="plugin_job.actor_principal_id")),
        )
        object.__setattr__(
            self,
            "target_representation_id",
            RepresentationId(
                validate_uuid(str(self.target_representation_id), field="plugin_job.target_representation_id")
            ),
        )
        object.__setattr__(self, "capability", require_non_empty(self.capability, field="plugin_job.capability"))
        if not isinstance(self.state, PluginJobState):
            object.__setattr__(self, "state", PluginJobState(self.state))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise DomainValidationError("plugin_job.revision must not be negative")
        if isinstance(self.progress, bool) or not isinstance(self.progress, (int, float)) or not 0 <= self.progress <= 1:
            raise DomainValidationError("plugin_job.progress must be between 0 and 1")
        object.__setattr__(self, "progress", float(self.progress))
        object.__setattr__(self, "created_at", as_utc(self.created_at, field="plugin_job.created_at"))
        object.__setattr__(self, "updated_at", as_utc(self.updated_at, field="plugin_job.updated_at"))
        if self.updated_at < self.created_at:
            raise DomainValidationError("plugin_job.updated_at cannot precede creation")
        terminal = self.state in {
            PluginJobState.SUCCEEDED,
            PluginJobState.FAILED,
            PluginJobState.CANCELLED,
        }
        if terminal != (self.finished_at is not None):
            raise DomainValidationError("terminal plugin job states require finished_at, non-terminal states forbid it")
        if self.finished_at is not None:
            finished_at = as_utc(self.finished_at, field="plugin_job.finished_at")
            if finished_at < self.updated_at:
                raise DomainValidationError("plugin_job.finished_at cannot precede its update")
            object.__setattr__(self, "finished_at", finished_at)
        if self.state is PluginJobState.SUCCEEDED and self.progress != 1.0:
            raise DomainValidationError("a succeeded plugin job must have progress 1")
        if self.error is not None:
            object.__setattr__(self, "error", require_non_empty(self.error, field="plugin_job.error", maximum=10_000))

    @classmethod
    def create(
        cls,
        *,
        project_id: ProjectId,
        layer_id: LayerId,
        selection: FrameSelectionV1,
        actor_principal_id: PrincipalId,
        target_representation_id: RepresentationId,
        capability: str,
        job_id: PluginJobId | str | None = None,
        created_at: datetime | None = None,
    ) -> PluginJob:
        now = created_at or utc_now()
        return cls(
            id=PluginJobId(str(job_id) if job_id is not None else new_uuid()),
            project_id=project_id,
            layer_id=layer_id,
            selection=selection,
            actor_principal_id=actor_principal_id,
            target_representation_id=target_representation_id,
            capability=capability,
            state=PluginJobState.QUEUED,
            revision=0,
            progress=0,
            created_at=now,
            updated_at=now,
        )

    def transition(
        self,
        state: PluginJobState,
        *,
        at: datetime | None = None,
        progress: float | None = None,
        error: str | None = None,
    ) -> PluginJob:
        state = PluginJobState(state)
        if state is self.state:
            return self
        if state not in PLUGIN_JOB_TRANSITIONS[self.state]:
            raise InvalidStateTransition(f"plugin job cannot transition from {self.state} to {state}")
        changed_at = as_utc(at or utc_now(), field="plugin_job.transition_at")
        if changed_at < self.updated_at:
            raise DomainValidationError("plugin job transition time cannot move backwards")
        next_progress = self.progress if progress is None else float(progress)
        if next_progress < self.progress and state not in {PluginJobState.QUEUED, PluginJobState.STAGING}:
            raise DomainValidationError("plugin job progress cannot move backwards")
        if state is PluginJobState.SUCCEEDED:
            next_progress = 1.0
        terminal = state in {PluginJobState.SUCCEEDED, PluginJobState.FAILED, PluginJobState.CANCELLED}
        return replace(
            self,
            state=state,
            revision=self.revision + 1,
            progress=next_progress,
            updated_at=changed_at,
            finished_at=changed_at if terminal else None,
            error=error,
        )


@dataclass(frozen=True, slots=True)
class PluginInputV1:
    frame_id: FrameId
    artifact_version_id: ArtifactVersionId
    sha256: str
    relative_path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", FrameId(validate_uuid(str(self.frame_id), field="plugin_input.frame_id")))
        object.__setattr__(
            self,
            "artifact_version_id",
            ArtifactVersionId(validate_uuid(str(self.artifact_version_id), field="plugin_input.artifact_version_id")),
        )
        object.__setattr__(self, "sha256", validate_sha256(self.sha256, field="plugin_input.sha256"))
        object.__setattr__(self, "relative_path", validate_relative_manifest_path(self.relative_path))


@dataclass(frozen=True, slots=True)
class PluginJobManifestV1:
    SCHEMA_VERSION: ClassVar[int] = 1

    job_id: PluginJobId
    project_id: ProjectId
    layer_id: LayerId
    target_representation_id: RepresentationId
    selection: FrameSelectionV1
    actor_principal_id: PrincipalId
    capability: str
    inputs: tuple[PluginInputV1, ...]
    parameters: Mapping[str, object]
    protocol_version: str = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", PluginJobId(validate_uuid(str(self.job_id), field="manifest.job_id")))
        object.__setattr__(
            self, "project_id", ProjectId(validate_uuid(str(self.project_id), field="manifest.project_id"))
        )
        object.__setattr__(self, "layer_id", LayerId(validate_uuid(str(self.layer_id), field="manifest.layer_id")))
        object.__setattr__(
            self,
            "target_representation_id",
            RepresentationId(
                validate_uuid(str(self.target_representation_id), field="manifest.target_representation_id")
            ),
        )
        object.__setattr__(
            self,
            "actor_principal_id",
            PrincipalId(validate_uuid(str(self.actor_principal_id), field="manifest.actor_principal_id")),
        )
        object.__setattr__(self, "capability", require_non_empty(self.capability, field="manifest.capability"))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if not self.inputs:
            raise DomainValidationError("plugin manifest requires at least one input")
        paths = [item.relative_path.casefold() for item in self.inputs]
        if len(paths) != len(set(paths)):
            raise DomainValidationError("plugin input paths must be unique, including on case-insensitive filesystems")
        frame_ids = [item.frame_id for item in self.inputs]
        if len(frame_ids) != len(set(frame_ids)):
            raise DomainValidationError("plugin manifest cannot contain multiple inputs for one frame")
        object.__setattr__(self, "parameters", freeze_mapping(self.parameters, field="manifest.parameters"))
        object.__setattr__(
            self, "protocol_version", require_non_empty(self.protocol_version, field="manifest.protocol_version")
        )


class PluginFrameOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class PluginResultOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class PluginFrameResultV1:
    output_id: str
    frame_id: FrameId
    outcome: PluginFrameOutcome
    relative_path: str | None = None
    sha256: str | None = None
    warning: str | None = None
    error: str | None = None
    media_type: str | None = None
    role: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_id", validate_uuid(self.output_id, field="plugin_result.output_id"))
        object.__setattr__(self, "frame_id", FrameId(validate_uuid(str(self.frame_id), field="plugin_result.frame_id")))
        if not isinstance(self.outcome, PluginFrameOutcome):
            object.__setattr__(self, "outcome", PluginFrameOutcome(self.outcome))
        has_file = self.relative_path is not None or self.sha256 is not None
        if self.outcome is PluginFrameOutcome.SUCCEEDED:
            if self.relative_path is None or self.sha256 is None:
                raise DomainValidationError("successful plugin frame result requires path and SHA-256")
        elif has_file:
            raise DomainValidationError("failed/skipped plugin frame result cannot declare output content")
        if self.relative_path is not None:
            object.__setattr__(self, "relative_path", validate_relative_manifest_path(self.relative_path))
        if self.sha256 is not None:
            object.__setattr__(self, "sha256", validate_sha256(self.sha256, field="plugin_result.sha256"))
        if self.outcome is PluginFrameOutcome.SUCCEEDED:
            media_type = require_non_empty(
                self.media_type or "", field="plugin_result.media_type", maximum=255
            ).lower()
            if "/" not in media_type:
                raise DomainValidationError("plugin_result.media_type must be a MIME type")
            object.__setattr__(self, "media_type", media_type)
            object.__setattr__(
                self,
                "role",
                require_non_empty(self.role or "", field="plugin_result.role", maximum=128),
            )
        elif self.media_type is not None or self.role is not None:
            raise DomainValidationError("failed/skipped plugin frame result cannot declare output metadata")
        if self.warning is not None:
            object.__setattr__(self, "warning", require_non_empty(self.warning, field="plugin_result.warning", maximum=10_000))
        if self.error is not None:
            object.__setattr__(self, "error", require_non_empty(self.error, field="plugin_result.error", maximum=10_000))


@dataclass(frozen=True, slots=True)
class PluginResultManifestV1:
    SCHEMA_VERSION: ClassVar[int] = 1

    job_id: PluginJobId
    plugin_name: str
    plugin_version: str
    results: tuple[PluginFrameResultV1, ...]
    parameters_applied: Mapping[str, object]
    protocol_version: str = "1.0"
    outcome: PluginResultOutcome = PluginResultOutcome.SUCCEEDED

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", PluginJobId(validate_uuid(str(self.job_id), field="result_manifest.job_id")))
        object.__setattr__(self, "plugin_name", require_non_empty(self.plugin_name, field="result_manifest.plugin_name"))
        object.__setattr__(
            self, "plugin_version", require_non_empty(self.plugin_version, field="result_manifest.plugin_version")
        )
        object.__setattr__(self, "results", tuple(self.results))
        if not isinstance(self.outcome, PluginResultOutcome):
            object.__setattr__(self, "outcome", PluginResultOutcome(self.outcome))
        if self.outcome in {PluginResultOutcome.FAILED, PluginResultOutcome.CANCELLED} and self.results:
            raise DomainValidationError("failed/cancelled plugin result cannot publish frame outputs")
        output_ids = [item.output_id for item in self.results]
        if len(output_ids) != len(set(output_ids)):
            raise DomainValidationError("plugin result output IDs must be unique")
        frame_ids = [item.frame_id for item in self.results]
        if len(frame_ids) != len(set(frame_ids)):
            raise DomainValidationError("plugin result cannot contain multiple outcomes for one frame")
        successful_paths = [
            item.relative_path.casefold()
            for item in self.results
            if item.relative_path is not None
        ]
        if len(successful_paths) != len(set(successful_paths)):
            raise DomainValidationError("plugin result paths must be unique")
        object.__setattr__(
            self,
            "parameters_applied",
            freeze_mapping(self.parameters_applied, field="result_manifest.parameters_applied"),
        )
        object.__setattr__(
            self,
            "protocol_version",
            require_non_empty(self.protocol_version, field="result_manifest.protocol_version"),
        )


class ReviewBatchState(StrEnum):
    DRAFT = "draft"
    ISSUED = "issued"
    PARTIALLY_RETURNED = "partially_returned"
    AWAITING_ACCEPTANCE = "awaiting_acceptance"
    COMPLETED = "completed"
    CHANGES_REQUESTED = "changes_requested"
    CANCELLED = "cancelled"


REVIEW_BATCH_TRANSITIONS: Final = {
    ReviewBatchState.DRAFT: frozenset({ReviewBatchState.ISSUED, ReviewBatchState.CANCELLED}),
    ReviewBatchState.ISSUED: frozenset(
        {
            ReviewBatchState.PARTIALLY_RETURNED,
            ReviewBatchState.AWAITING_ACCEPTANCE,
            ReviewBatchState.COMPLETED,
            ReviewBatchState.CANCELLED,
        }
    ),
    ReviewBatchState.PARTIALLY_RETURNED: frozenset(
        {ReviewBatchState.AWAITING_ACCEPTANCE, ReviewBatchState.COMPLETED, ReviewBatchState.CANCELLED}
    ),
    ReviewBatchState.AWAITING_ACCEPTANCE: frozenset(
        {ReviewBatchState.COMPLETED, ReviewBatchState.CHANGES_REQUESTED}
    ),
    ReviewBatchState.CHANGES_REQUESTED: frozenset(
        {ReviewBatchState.ISSUED, ReviewBatchState.CANCELLED}
    ),
    ReviewBatchState.COMPLETED: frozenset(),
    ReviewBatchState.CANCELLED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class ReviewItem:
    frame_id: FrameId
    vector_version_id: ArtifactVersionId
    vector_sha256: str
    image_version_id: ArtifactVersionId | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", FrameId(validate_uuid(str(self.frame_id), field="review_item.frame_id")))
        object.__setattr__(
            self,
            "vector_version_id",
            ArtifactVersionId(validate_uuid(str(self.vector_version_id), field="review_item.vector_version_id")),
        )
        object.__setattr__(
            self, "vector_sha256", validate_sha256(self.vector_sha256, field="review_item.vector_sha256")
        )
        if self.image_version_id is not None:
            object.__setattr__(
                self,
                "image_version_id",
                ArtifactVersionId(validate_uuid(str(self.image_version_id), field="review_item.image_version_id")),
            )


@dataclass(frozen=True, slots=True)
class ReviewBatch:
    id: ReviewBatchId
    project_id: ProjectId
    layer_id: LayerId
    selection: FrameSelectionV1
    items: tuple[ReviewItem, ...]
    assignee_id: PerformerId
    created_by: PrincipalId
    instructions: str
    state: ReviewBatchState
    revision: int
    created_at: datetime
    updated_at: datetime
    due_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", ReviewBatchId(validate_uuid(str(self.id), field="review_batch.id")))
        object.__setattr__(
            self, "project_id", ProjectId(validate_uuid(str(self.project_id), field="review_batch.project_id"))
        )
        object.__setattr__(
            self, "layer_id", LayerId(validate_uuid(str(self.layer_id), field="review_batch.layer_id"))
        )
        object.__setattr__(self, "items", tuple(self.items))
        if not self.items:
            raise DomainValidationError("review batch requires at least one versioned item")
        item_frames = [item.frame_id for item in self.items]
        if len(item_frames) != len(set(item_frames)):
            raise DomainValidationError("review batch cannot contain a frame more than once")
        object.__setattr__(
            self, "assignee_id", PerformerId(validate_uuid(str(self.assignee_id), field="review_batch.assignee_id"))
        )
        object.__setattr__(
            self,
            "created_by",
            PrincipalId(validate_uuid(str(self.created_by), field="review_batch.created_by")),
        )
        object.__setattr__(self, "instructions", self.instructions.strip())
        if not isinstance(self.state, ReviewBatchState):
            object.__setattr__(self, "state", ReviewBatchState(self.state))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise DomainValidationError("review_batch.revision must not be negative")
        object.__setattr__(self, "created_at", as_utc(self.created_at, field="review_batch.created_at"))
        object.__setattr__(self, "updated_at", as_utc(self.updated_at, field="review_batch.updated_at"))
        if self.updated_at < self.created_at:
            raise DomainValidationError("review_batch.updated_at cannot precede creation")
        if self.due_at is not None:
            due_at = as_utc(self.due_at, field="review_batch.due_at")
            if due_at < self.created_at:
                raise DomainValidationError("review batch due date cannot precede creation")
            object.__setattr__(self, "due_at", due_at)

    @classmethod
    def create(
        cls,
        *,
        project_id: ProjectId,
        layer_id: LayerId,
        selection: FrameSelectionV1,
        items: tuple[ReviewItem, ...],
        assignee_id: PerformerId,
        created_by: PrincipalId,
        instructions: str = "",
        due_at: datetime | None = None,
        batch_id: ReviewBatchId | str | None = None,
        created_at: datetime | None = None,
    ) -> ReviewBatch:
        now = created_at or utc_now()
        return cls(
            id=ReviewBatchId(str(batch_id) if batch_id is not None else new_uuid()),
            project_id=project_id,
            layer_id=layer_id,
            selection=selection,
            items=items,
            assignee_id=assignee_id,
            created_by=created_by,
            instructions=instructions,
            state=ReviewBatchState.DRAFT,
            revision=0,
            created_at=now,
            updated_at=now,
            due_at=due_at,
        )

    def transition(self, state: ReviewBatchState, *, at: datetime | None = None) -> ReviewBatch:
        state = ReviewBatchState(state)
        if state is self.state:
            return self
        if state not in REVIEW_BATCH_TRANSITIONS[self.state]:
            raise InvalidStateTransition(f"review batch cannot transition from {self.state} to {state}")
        changed_at = as_utc(at or utc_now(), field="review_batch.transition_at")
        if changed_at < self.updated_at:
            raise DomainValidationError("review batch transition time cannot move backwards")
        return replace(self, state=state, revision=self.revision + 1, updated_at=changed_at)

    def issue(self, *, at: datetime | None = None) -> ReviewBatch:
        return self.transition(ReviewBatchState.ISSUED, at=at)

    def register_return(
        self,
        *,
        has_missing: bool,
        has_changed: bool,
        at: datetime | None = None,
    ) -> ReviewBatch:
        if self.state not in {ReviewBatchState.ISSUED, ReviewBatchState.PARTIALLY_RETURNED}:
            raise InvalidStateTransition("only an issued review batch can receive returned files")
        if has_missing:
            target = ReviewBatchState.PARTIALLY_RETURNED
        elif has_changed:
            target = ReviewBatchState.AWAITING_ACCEPTANCE
        else:
            target = ReviewBatchState.COMPLETED
        # A later return may still be partial while carrying newly returned
        # files.  It is a real aggregate change even though the visible state
        # remains ``partially_returned`` and therefore needs its own revision.
        if target is self.state:
            changed_at = as_utc(at or utc_now(), field="review_batch.return_at")
            if changed_at < self.updated_at:
                raise DomainValidationError("review batch return time cannot move backwards")
            return replace(self, revision=self.revision + 1, updated_at=changed_at)
        return self.transition(target, at=at)


@dataclass(frozen=True, slots=True)
class ReviewPackageFileV1:
    frame_id: FrameId
    artifact_version_id: ArtifactVersionId
    sha256: str
    relative_path: str
    x: int | None = None
    y: int | None = None
    role: str = "vector"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", FrameId(validate_uuid(str(self.frame_id), field="review_file.frame_id")))
        object.__setattr__(
            self,
            "artifact_version_id",
            ArtifactVersionId(validate_uuid(str(self.artifact_version_id), field="review_file.artifact_version_id")),
        )
        object.__setattr__(self, "sha256", validate_sha256(self.sha256, field="review_file.sha256"))
        object.__setattr__(self, "relative_path", validate_relative_manifest_path(self.relative_path))
        if (self.x is None) != (self.y is None):
            raise DomainValidationError("review file coordinates must contain both X and Y")
        if self.x is not None and (self.x < 1 or self.y is None or self.y < 1):
            raise DomainValidationError("review file coordinates are one-based")
        role = require_non_empty(self.role, field="review_file.role", maximum=32).lower()
        if role not in {"vector", "image"}:
            raise DomainValidationError("review file role must be vector or image")
        object.__setattr__(self, "role", role)


@dataclass(frozen=True, slots=True)
class ReviewPackageManifestV1:
    SCHEMA_VERSION: ClassVar[int] = 1

    package_id: ReviewBatchId
    project_id: ProjectId
    layer_id: LayerId
    issued_at: datetime
    files: tuple[ReviewPackageFileV1, ...]
    signature_algorithm: str = "ed25519"
    batch_id: ReviewBatchId | None = None
    performer_id: PerformerId | None = None
    issued_by: PrincipalId | None = None
    due_at: datetime | None = None
    instructions: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "package_id", ReviewBatchId(validate_uuid(str(self.package_id), field="review_manifest.package_id"))
        )
        object.__setattr__(
            self, "project_id", ProjectId(validate_uuid(str(self.project_id), field="review_manifest.project_id"))
        )
        object.__setattr__(
            self, "layer_id", LayerId(validate_uuid(str(self.layer_id), field="review_manifest.layer_id"))
        )
        object.__setattr__(self, "issued_at", as_utc(self.issued_at, field="review_manifest.issued_at"))
        object.__setattr__(self, "files", tuple(self.files))
        if not self.files:
            raise DomainValidationError("review package manifest requires files")
        paths = [item.relative_path.casefold() for item in self.files]
        if len(paths) != len(set(paths)):
            raise DomainValidationError("review package paths must be unique")
        frame_roles = [(item.frame_id, item.role) for item in self.files]
        if len(frame_roles) != len(set(frame_roles)):
            raise DomainValidationError("review package cannot repeat a file role for one frame")
        if self.signature_algorithm.lower() != "ed25519":
            raise DomainValidationError("V1 review manifests must use Ed25519 signatures")
        object.__setattr__(self, "signature_algorithm", "ed25519")
        if self.batch_id is not None:
            object.__setattr__(
                self,
                "batch_id",
                ReviewBatchId(validate_uuid(str(self.batch_id), field="review_manifest.batch_id")),
            )
        if self.performer_id is not None:
            object.__setattr__(
                self,
                "performer_id",
                PerformerId(validate_uuid(str(self.performer_id), field="review_manifest.performer_id")),
            )
        if self.issued_by is not None:
            object.__setattr__(
                self,
                "issued_by",
                PrincipalId(validate_uuid(str(self.issued_by), field="review_manifest.issued_by")),
            )
        if self.due_at is not None:
            object.__setattr__(self, "due_at", as_utc(self.due_at, field="review_manifest.due_at"))
        object.__setattr__(self, "instructions", self.instructions.strip())


class ReviewFileStatus(StrEnum):
    UNCHANGED = "unchanged"
    CHANGED = "changed"
    MISSING = "missing"
    EXTRA = "extra"
    DUPLICATE = "duplicate"
    INVALID = "invalid"
    STALE_BASE_CONFLICT = "stale_base_conflict"


@dataclass(frozen=True, slots=True)
class ReviewFileComparison:
    status: ReviewFileStatus
    relative_path: str
    frame_id: FrameId | None = None
    expected_version_id: ArtifactVersionId | None = None
    expected_sha256: str | None = None
    returned_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ReviewFileStatus):
            object.__setattr__(self, "status", ReviewFileStatus(self.status))
        if self.status is ReviewFileStatus.INVALID:
            object.__setattr__(
                self,
                "relative_path",
                require_non_empty(self.relative_path, field="comparison.relative_path", maximum=2048),
            )
        else:
            object.__setattr__(self, "relative_path", validate_relative_manifest_path(self.relative_path))
        if self.frame_id is not None:
            object.__setattr__(self, "frame_id", FrameId(validate_uuid(str(self.frame_id), field="comparison.frame_id")))
        if self.expected_version_id is not None:
            object.__setattr__(
                self,
                "expected_version_id",
                ArtifactVersionId(
                    validate_uuid(str(self.expected_version_id), field="comparison.expected_version_id")
                ),
            )
        if self.expected_sha256 is not None:
            object.__setattr__(
                self, "expected_sha256", validate_sha256(self.expected_sha256, field="comparison.expected_sha256")
            )
        if self.returned_sha256 is not None:
            object.__setattr__(
                self, "returned_sha256", validate_sha256(self.returned_sha256, field="comparison.returned_sha256")
            )


@dataclass(frozen=True, slots=True)
class ReviewComparisonReport:
    items: tuple[ReviewFileComparison, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))

    def count(self, status: ReviewFileStatus) -> int:
        return sum(item.status is status for item in self.items)

    @property
    def can_commit(self) -> bool:
        return not any(item.status in {ReviewFileStatus.DUPLICATE, ReviewFileStatus.INVALID} for item in self.items)


__all__ = [
    "FrameStatus",
    "FrameStatusFacts",
    "PLUGIN_JOB_TRANSITIONS",
    "REVIEW_BATCH_TRANSITIONS",
    "PluginFrameOutcome",
    "PluginFrameResultV1",
    "PluginInputV1",
    "PluginJob",
    "PluginJobManifestV1",
    "PluginJobState",
    "PluginResultManifestV1",
    "PluginResultOutcome",
    "ReviewBatch",
    "ReviewBatchState",
    "ReviewComparisonReport",
    "ReviewFileComparison",
    "ReviewFileStatus",
    "ReviewItem",
    "ReviewPackageFileV1",
    "ReviewPackageManifestV1",
    "validate_relative_manifest_path",
    "derive_frame_status",
]
