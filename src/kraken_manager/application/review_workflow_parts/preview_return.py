"""review_workflow: preview return."""
from __future__ import annotations
from typing import Any
from kraken_manager.application.dto import (
    CommitReviewReturnCommand,
    ReturnedFileDigest,
    ReviewReturnCommitResult,
)
from kraken_manager.application.errors import ConflictError
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.workflows import (
    ReviewBatchState,
    ReviewFileStatus,
)

from kraken_manager.application.review_workflow_parts.review_handler import (
    _prior_command_event,
    _review_stream,
)


class PreviewReturnMixin:
    def _preview_return(
        self,
        command: CommitReviewReturnCommand,
        prepared: tuple[Any, tuple[ReturnedFileDigest, ...], list[tuple[str, str | None, bool]]],
    ) -> ReviewReturnCommitResult | tuple[tuple[str, str], ...]:
        with self._uow_factory() as uow:
            batch = self._load_batch(
                uow,
                project_id=command.project_id,
                batch_id=command.batch_id,
                context=command.context,
                permission=Permission.RETURN_REVIEW,
            )
            committed = _prior_command_event(
                uow,
                project_id=batch.project_id,
                idempotency_key=command.context.idempotency_key,
                event_type="ReviewReturnCommitted",
                batch_id=batch.id,
            )
            if committed is not None:
                return self._result_from_event(uow, batch, committed)
            plan = self._inspect(uow, batch, command.source, prepared)
            review_events = uow.event_store.load_stream(_review_stream(batch.id))
            for event in reversed(review_events):
                if (
                    event.event_type == "ReviewReturnCommitted"
                    and event.payload.get("return_fingerprint") == plan.return_fingerprint
                    and str(event.payload.get("package_id")) == str(plan.package_id)
                ):
                    return self._result_from_event(uow, batch, event)
            self._require_revision(uow, batch, command.expected_batch_revision)
            if batch.state not in {ReviewBatchState.ISSUED, ReviewBatchState.PARTIALLY_RETURNED}:
                raise ConflictError("Only an issued review batch can receive returned files")
            if not plan.report.can_commit:
                raise ConflictError("Duplicate or invalid returned files must be resolved before commit")
            manifest = prepared[0]
            expected_by_path = {
                item.relative_path.casefold(): item for item in manifest.files if item.role == "vector"
            }
            writes: list[tuple[str, str]] = []
            for comparison in plan.report.items:
                if comparison.status is ReviewFileStatus.EXTRA and comparison.returned_sha256 is not None:
                    writes.append((comparison.relative_path, comparison.returned_sha256))
            for comparison in plan.report.items:
                if comparison.status not in {
                    ReviewFileStatus.CHANGED,
                    ReviewFileStatus.STALE_BASE_CONFLICT,
                }:
                    continue
                if comparison.returned_sha256 is None:
                    continue
                if (
                    comparison.status is ReviewFileStatus.CHANGED
                    and comparison.returned_sha256 == comparison.expected_sha256
                ):
                    continue
                if expected_by_path.get(comparison.relative_path.casefold()) is None:
                    continue
                writes.append((comparison.relative_path, comparison.returned_sha256))
            return tuple(writes)
