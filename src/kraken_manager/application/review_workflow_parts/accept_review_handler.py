"""review_workflow: accept review handler."""
from __future__ import annotations
from kraken_manager.application.dto import (
    AcceptReviewCommand,
)
from kraken_manager.application.errors import ConflictError
from kraken_manager.application.use_cases import (
    _series_stream,
)
from kraken_manager.domain.artifacts import ArtifactVersion
from kraken_manager.domain.common import ArtifactVersionId
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

class AcceptReviewHandler(_ReviewHandler):
    def __call__(self, command: AcceptReviewCommand) -> ReviewBatch:
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
                event_type="ReviewBatchAccepted",
                batch_id=batch.id,
            ) is not None:
                return batch
            self._require_revision(uow, batch, command.expected_batch_revision)
            if batch.state is not ReviewBatchState.AWAITING_ACCEPTANCE:
                raise ConflictError("Review batch is not awaiting acceptance")
            requested_ids = tuple(dict.fromkeys(command.candidate_version_ids))
            if not requested_ids:
                raise ConflictError("Acceptance requires candidate versions")

            latest_by_series: dict[str, ArtifactVersionId] = {}
            recorded_ids: set[ArtifactVersionId] = set()
            for event in uow.event_store.load_stream(_review_stream(batch.id)):
                if event.event_type != "ReviewReturnCommitted":
                    continue
                for value in event.payload.get("candidate_version_ids", ()):
                    identifier = ArtifactVersionId(str(value))
                    candidate = uow.projections.get_artifact_version(identifier)
                    if candidate is not None:
                        recorded_ids.add(identifier)
                        latest_by_series[str(candidate.series_id)] = identifier
            required_ids = set(latest_by_series.values())
            if set(requested_ids) != required_ids or not set(requested_ids) <= recorded_ids:
                raise ConflictError("All latest candidates from this review batch must be accepted together")

            base_ids = {item.vector_version_id for item in batch.items}
            candidates: list[ArtifactVersion] = []
            series_ids: set[str] = set()
            for identifier in requested_ids:
                candidate = uow.projections.get_artifact_version(identifier)
                if candidate is None or candidate.parent_version_id not in base_ids:
                    raise ConflictError("Candidate does not belong to this review batch")
                if str(candidate.series_id) in series_ids:
                    raise ConflictError("Only one candidate per artifact series may be accepted")
                series_ids.add(str(candidate.series_id))
                candidates.append(candidate)

            now = self._clock.now()
            for candidate in candidates:
                stream_id = _series_stream(str(candidate.series_id))
                revision = uow.event_store.current_revision(stream_id)
                activation = EventEnvelope.create(
                    stream_id=stream_id,
                    project_id=batch.project_id,
                    revision=revision + 1,
                    event_type="ArtifactVersionActivated",
                    payload={
                        "artifact_version_id": candidate.id,
                        "series_id": candidate.series_id,
                        "review_batch_id": batch.id,
                    },
                    actor=ActorSnapshot.from_principal(command.context.actor),
                    recorded_at=now,
                    effective_at=command.context.effective_at,
                    performer_id=batch.assignee_id,
                    correlation_id=command.context.correlation_id,
                )
                uow.event_store.append(stream_id, expected_revision=revision, events=(activation,))
                uow.projections.save_artifact_version(candidate, activate=True)

            accepted = batch.transition(ReviewBatchState.COMPLETED, at=now)
            event = EventEnvelope.create(
                stream_id=_review_stream(batch.id),
                project_id=batch.project_id,
                revision=accepted.revision,
                event_type="ReviewBatchAccepted",
                payload={
                    "review_batch_id": batch.id,
                    "state": accepted.state.value,
                    "batch_revision": accepted.revision,
                    "updated_at": accepted.updated_at.isoformat(),
                    "candidate_version_ids": list(requested_ids),
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
            uow.projections.save_review_batch(accepted)
            uow.commit()
            return accepted
