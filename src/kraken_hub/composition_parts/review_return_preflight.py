"""composition: review return preflight."""
from __future__ import annotations
from collections.abc import Iterable
from pathlib import Path
from kraken_manager.application.dto import (
    AcceptReviewCommand,
    CancelReviewBatchCommand,
    CommitReviewReturnCommand,
    CommandContext,
    DryRunReviewReturnCommand,
    ExportReviewPackageCommand,
    RequestReviewChangesCommand,
)
from kraken_manager.application.review_workflow import (
    AcceptReviewHandler,
    CancelReviewBatchHandler,
    CommitReviewReturnHandler,
    DryRunReviewReturnHandler,
    ExportReviewPackageHandler,
    RequestReviewChangesHandler,
)
from kraken_manager.domain.common import (
    ArtifactVersionId,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.workflows import (
    ReviewBatch,
)
from kraken_manager.infrastructure.review import ReviewPackageReader, ReviewPackageWriter


class ReviewReturnPreflightMixin:
    def export_review_batch(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        destination: Path | str,
        idempotency_key: str,
    ) -> ReviewBatch:
        writer = ReviewPackageWriter(self._review_keys().private_key)
        return ExportReviewPackageHandler(
            self._uow(str(batch.project_id)),
            self.profiles,
            self.clock,
            writer,
        )(
            ExportReviewPackageCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=batch.project_id,
                batch_id=batch.id,
                expected_batch_revision=batch.revision,
                destination=str(destination),
                include_images=True,
            )
        )

    def review_return_preflight(
        self,
        *,
        principal: Principal,
        source: Path | str,
        idempotency_key: str,
    ):
        reader = ReviewPackageReader(self._review_keys().public_key)
        manifest = reader.read_manifest(source)
        batch_id = manifest.batch_id or manifest.package_id
        projection = self._projection(manifest.project_id)
        batch = projection.get_review_batch(batch_id)
        if batch is None:
            raise ValueError("Manifest does not match a known review batch")
        plan = DryRunReviewReturnHandler(
            self._uow(str(manifest.project_id)),
            self.profiles,
            self.clock,
            reader,
        )(
            DryRunReviewReturnCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=manifest.project_id,
                batch_id=batch.id,
                expected_batch_revision=batch.revision,
                source=str(source),
            )
        )
        return batch, plan

    def commit_review_return(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        source: Path | str,
        idempotency_key: str,
    ):
        reader = ReviewPackageReader(self._review_keys().public_key)
        return CommitReviewReturnHandler(
            self._uow(str(batch.project_id)),
            self.profiles,
            self.clock,
            reader,
        )(
            CommitReviewReturnCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=batch.project_id,
                batch_id=batch.id,
                expected_batch_revision=batch.revision,
                source=str(source),
            )
        )

    def accept_review(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        candidate_version_ids: Iterable[ArtifactVersionId | str],
        idempotency_key: str,
    ) -> ReviewBatch:
        return AcceptReviewHandler(
            self._uow(str(batch.project_id)),
            self.profiles,
            self.clock,
        )(
            AcceptReviewCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=batch.project_id,
                batch_id=batch.id,
                expected_batch_revision=batch.revision,
                candidate_version_ids=tuple(
                    ArtifactVersionId(str(value))
                    for value in candidate_version_ids
                ),
            )
        )

    def request_review_changes(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        reason: str,
        idempotency_key: str,
    ) -> ReviewBatch:
        return RequestReviewChangesHandler(
            self._uow(str(batch.project_id)),
            self.profiles,
            self.clock,
        )(
            RequestReviewChangesCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=batch.project_id,
                batch_id=batch.id,
                expected_batch_revision=batch.revision,
                reason=reason,
            )
        )

    def cancel_review_batch(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        idempotency_key: str,
    ) -> ReviewBatch:
        return CancelReviewBatchHandler(
            self._uow(str(batch.project_id)),
            self.profiles,
            self.clock,
        )(
            CancelReviewBatchCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=batch.project_id,
                batch_id=batch.id,
                expected_batch_revision=batch.revision,
            )
        )

    def review_candidate_version_ids(
        self,
        batch: ReviewBatch,
    ) -> tuple[ArtifactVersionId, ...]:
        latest_by_series: dict[str, ArtifactVersionId] = {}
        projection = self._projection(batch.project_id)
        for event in self.history(batch.project_id):
            if (
                event.event_type != "ReviewReturnCommitted"
                or str(event.payload.get("review_batch_id", "")) != str(batch.id)
            ):
                continue
            for value in event.payload.get("candidate_version_ids", ()):
                identifier = ArtifactVersionId(str(value))
                version = projection.get_artifact_version(identifier)
                if version is not None:
                    latest_by_series[str(version.series_id)] = identifier
        return tuple(latest_by_series[key] for key in sorted(latest_by_series))
