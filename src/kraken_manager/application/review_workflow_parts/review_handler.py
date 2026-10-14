"""review_workflow: review handler."""
from __future__ import annotations
from collections.abc import Mapping
from kraken_manager.application.errors import ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.application.ports import (
    UnitOfWork,
)
from kraken_manager.application.use_cases import (
    _ProjectHandler,
)
from kraken_manager.domain.artifacts import ArtifactVersion
from kraken_manager.domain.common import ArtifactVersionId, ReviewBatchId
from kraken_manager.domain.events import EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.workflows import (
    ReviewBatch,
    ReviewComparisonReport,
    ReviewFileComparison,
    ReviewFileStatus,
    ReviewItem,
)

def _review_stream(batch_id: ReviewBatchId | str) -> str:
    return f"review-batch:{batch_id}"

def _prior_command_event(
    uow: UnitOfWork,
    *,
    project_id,
    idempotency_key: str,
    event_type: str,
    batch_id: ReviewBatchId | None = None,
) -> EventEnvelope | None:
    events = uow.event_store.find_by_idempotency_key(project_id, idempotency_key)
    matches = [
        event
        for event in events
        if event.event_type == event_type
        and (batch_id is None or str(event.payload.get("review_batch_id")) == str(batch_id))
    ]
    if matches:
        return matches[-1]
    if events:
        raise ConflictError("Idempotency key was already used by another command")
    return None

def _review_item_payload(item: ReviewItem) -> dict[str, object]:
    return {
        "frame_id": item.frame_id,
        "vector_version_id": item.vector_version_id,
        "vector_sha256": item.vector_sha256,
        "image_version_id": item.image_version_id,
    }

def _batch_payload(batch: ReviewBatch) -> dict[str, object]:
    return {
        "review_batch_id": batch.id,
        "layer_id": batch.layer_id,
        "selection": batch.selection.to_dict(),
        "items": [_review_item_payload(item) for item in batch.items],
        "assignee_id": batch.assignee_id,
        "created_by": batch.created_by,
        "instructions": batch.instructions,
        "state": batch.state.value,
        "batch_revision": batch.revision,
        "created_at": batch.created_at.isoformat(),
        "updated_at": batch.updated_at.isoformat(),
        "due_at": None if batch.due_at is None else batch.due_at.isoformat(),
    }

def _comparison_payload(comparison: ReviewFileComparison) -> dict[str, object]:
    return {
        "status": comparison.status.value,
        "relative_path": comparison.relative_path,
        "frame_id": comparison.frame_id,
        "expected_version_id": comparison.expected_version_id,
        "expected_sha256": comparison.expected_sha256,
        "returned_sha256": comparison.returned_sha256,
    }

def _report_from_payload(payload: object) -> ReviewComparisonReport:
    if not isinstance(payload, (tuple, list)):
        raise ConflictError("Stored review return has an invalid comparison report")
    return ReviewComparisonReport(
        tuple(
            ReviewFileComparison(
                status=ReviewFileStatus(str(item["status"])),
                relative_path=str(item["relative_path"]),
                frame_id=None if item.get("frame_id") is None else str(item["frame_id"]),
                expected_version_id=None
                if item.get("expected_version_id") is None
                else ArtifactVersionId(str(item["expected_version_id"])),
                expected_sha256=None if item.get("expected_sha256") is None else str(item["expected_sha256"]),
                returned_sha256=None if item.get("returned_sha256") is None else str(item["returned_sha256"]),
            )
            for item in payload
            if isinstance(item, Mapping)
        )
    )

def _artifact_created_payload(version: ArtifactVersion, *, activated: bool, candidate: bool) -> dict[str, object]:
    if version.blob is None:
        raise ConflictError("A managed artifact version is required")
    return {
        "artifact_version_id": version.id,
        "series_id": version.series_id,
        "sha256": version.sha256,
        "size_bytes": version.size_bytes,
        "media_type": version.media_type,
        "filename": version.filename,
        "blob": {"sha256": version.blob.sha256, "size_bytes": version.blob.size_bytes},
        "parent_version_id": version.parent_version_id,
        "input_version_ids": version.input_version_ids,
        "tool_name": version.tool_name,
        "tool_version": version.tool_version,
        "parameters": version.parameters,
        "author_principal_id": version.author_principal_id,
        "created_at": version.created_at.isoformat(),
        "activated": activated,
        "candidate": candidate,
    }

class _ReviewHandler(_ProjectHandler):
    def _load_batch(
        self,
        uow: UnitOfWork,
        *,
        project_id,
        batch_id: ReviewBatchId,
        context,
        permission: Permission,
    ) -> ReviewBatch:
        project, _ = self._load_project_and_authorize(
            uow,
            project_id=project_id,
            actor=context.actor,
            permission=permission,
            gitlab_identity_verified=context.gitlab_identity_verified,
        )
        batch = uow.projections.get_review_batch(batch_id)
        if batch is None or batch.project_id != project.id:
            raise NotFoundError("Review batch was not found in the project")
        return batch

    @staticmethod
    def _require_revision(uow: UnitOfWork, batch: ReviewBatch, expected: int) -> None:
        if batch.revision != expected:
            raise ConcurrencyError(f"Expected review batch revision {expected}, found {batch.revision}")
        stream_revision = uow.event_store.current_revision(_review_stream(batch.id))
        if stream_revision != batch.revision:
            raise ConcurrencyError(
                f"Review batch projection is at revision {batch.revision}, stream is at {stream_revision}"
            )
