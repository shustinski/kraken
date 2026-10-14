"""review_workflow: export review package handler."""
from __future__ import annotations
from collections.abc import Callable, Iterator
from dataclasses import replace
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    ExportReviewPackageCommand,
    ReviewPackagePlan,
)
from kraken_manager.application.errors import ConflictError
from kraken_manager.application.ports import (
    Clock,
    ReviewPackageWriter,
    StorageProfileCatalog,
    UnitOfWork,
    UnitOfWorkFactory,
)
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.workflows import (
    ReviewBatch,
    ReviewBatchState,
)

from kraken_manager.application.review_workflow_parts.review_handler import _review_stream

from kraken_manager.application.review_workflow_parts.review_package_planner import _ReviewPackagePlanner

class ExportReviewPackageHandler(_ReviewPackagePlanner):
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        storage_profiles: StorageProfileCatalog,
        clock: Clock,
        writer: ReviewPackageWriter,
        authorization: AuthorizationPolicy | None = None,
    ) -> None:
        super().__init__(uow_factory, storage_profiles, clock, authorization)
        self._writer = writer

    def __call__(self, command: ExportReviewPackageCommand) -> ReviewBatch:
        with self._uow_factory() as uow:
            batch, plan, readers = self._load_export(uow, command)
            if plan is None or readers is None:
                return batch
        self._writer.write(command.destination, plan.manifest, readers)
        with self._uow_factory() as uow:
            batch, current, _readers = self._load_export(uow, command)
            if current is None:
                return batch
            now = plan.manifest.issued_at
            first_issue = batch.state in {
                ReviewBatchState.DRAFT,
                ReviewBatchState.CHANGES_REQUESTED,
            }
            issued = (
                batch.issue(at=now)
                if first_issue
                else replace(
                    batch,
                    revision=batch.revision + 1,
                    updated_at=now,
                )
            )
            event_type = "ReviewBatchIssued" if first_issue else "ReviewBatchReexported"
            event = EventEnvelope.create(
                stream_id=_review_stream(batch.id),
                project_id=batch.project_id,
                revision=issued.revision,
                event_type=event_type,
                payload={
                    "review_batch_id": batch.id,
                    "package_id": plan.manifest.package_id,
                    "state": issued.state.value,
                    "batch_revision": issued.revision,
                    "updated_at": issued.updated_at.isoformat(),
                    "file_count": len(plan.manifest.files),
                    "total_size_bytes": plan.total_size_bytes,
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
            uow.projections.save_review_batch(issued)
            uow.commit()
            return issued

    def _load_export(
        self, uow: UnitOfWork, command: ExportReviewPackageCommand
    ) -> tuple[ReviewBatch, ReviewPackagePlan | None, dict[str, Callable[[], Iterator[bytes]]] | None]:
        batch = self._load_batch(
            uow,
            project_id=command.project_id,
            batch_id=command.batch_id,
            context=command.context,
            permission=Permission.MANAGE_REVIEW,
        )
        prior_events = uow.event_store.find_by_idempotency_key(
            batch.project_id,
            command.context.idempotency_key,
        )
        if any(
            event.event_type in {"ReviewBatchIssued", "ReviewBatchReexported"}
            and str(event.payload.get("review_batch_id")) == str(batch.id)
            for event in prior_events
        ):
            return batch, None, None
        if prior_events:
            raise ConflictError("Idempotency key was already used by another command")
        self._require_revision(uow, batch, command.expected_batch_revision)
        if batch.state in {ReviewBatchState.COMPLETED, ReviewBatchState.CANCELLED}:
            raise ConflictError("A closed review batch cannot be exported")
        plan, readers = self._build_plan(
            uow,
            batch,
            package_id=command.package_id,
            issued_by=command.context.actor.id,
            include_images=command.include_images,
            issued_at=self._clock.now(),
        )
        if not plan.can_export:
            raise ConflictError("Review package preflight failed: " + "; ".join(plan.issues))
        return batch, plan, readers
