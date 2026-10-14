"""review_workflow: request review changes handler."""
from __future__ import annotations
from kraken_manager.application.dto import (
    CancelReviewBatchCommand,
    RequestReviewChangesCommand,
)
from kraken_manager.application.errors import ConflictError
from kraken_manager.domain.common import require_non_empty
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.workflows import (
    ReviewBatch,
    ReviewBatchState,
)

from kraken_manager.application.review_workflow_parts.review_handler import (
    _ReviewHandler,
    _prior_command_event,
    _review_stream,
)

class RequestReviewChangesHandler(_ReviewHandler):
    def __call__(self, command: RequestReviewChangesCommand) -> ReviewBatch:
        reason = require_non_empty(command.reason, field="review_changes.reason", maximum=100_000)
        with self._uow_factory() as uow:
            batch = self._load_batch(
                uow,
                project_id=command.project_id,
                batch_id=command.batch_id,
                context=command.context,
                permission=Permission.ACCEPT_REVIEW,
            )
            if _prior_command_event(
                uow,
                project_id=batch.project_id,
                idempotency_key=command.context.idempotency_key,
                event_type="ReviewChangesRequested",
                batch_id=batch.id,
            ) is not None:
                return batch
            self._require_revision(uow, batch, command.expected_batch_revision)
            if batch.state is not ReviewBatchState.AWAITING_ACCEPTANCE:
                raise ConflictError("Review batch is not awaiting acceptance")
            now = self._clock.now()
            changed = batch.transition(ReviewBatchState.CHANGES_REQUESTED, at=now)
            event = EventEnvelope.create(
                stream_id=_review_stream(batch.id),
                project_id=batch.project_id,
                revision=changed.revision,
                event_type="ReviewChangesRequested",
                payload={
                    "review_batch_id": batch.id,
                    "state": changed.state.value,
                    "batch_revision": changed.revision,
                    "updated_at": changed.updated_at.isoformat(),
                    "reason": reason,
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=batch.assignee_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(
                _review_stream(batch.id), expected_revision=batch.revision, events=(event,)
            )
            uow.projections.save_review_batch(changed)
            uow.commit()
            return changed

class CancelReviewBatchHandler(_ReviewHandler):
    def __call__(self, command: CancelReviewBatchCommand) -> ReviewBatch:
        with self._uow_factory() as uow:
            batch = self._load_batch(
                uow,
                project_id=command.project_id,
                batch_id=command.batch_id,
                context=command.context,
                permission=Permission.MANAGE_REVIEW,
            )
            if _prior_command_event(
                uow,
                project_id=batch.project_id,
                idempotency_key=command.context.idempotency_key,
                event_type="ReviewBatchCancelled",
                batch_id=batch.id,
            ) is not None:
                return batch
            self._require_revision(uow, batch, command.expected_batch_revision)
            if batch.state in {ReviewBatchState.COMPLETED, ReviewBatchState.CANCELLED}:
                raise ConflictError("Review batch is already closed")
            now = self._clock.now()
            cancelled = batch.transition(ReviewBatchState.CANCELLED, at=now)
            event = EventEnvelope.create(
                stream_id=_review_stream(batch.id),
                project_id=batch.project_id,
                revision=cancelled.revision,
                event_type="ReviewBatchCancelled",
                payload={
                    "review_batch_id": batch.id,
                    "state": cancelled.state.value,
                    "batch_revision": cancelled.revision,
                    "updated_at": cancelled.updated_at.isoformat(),
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=batch.assignee_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(
                _review_stream(batch.id),
                expected_revision=batch.revision,
                events=(event,),
            )
            uow.projections.save_review_batch(cancelled)
            uow.commit()
            return cancelled
