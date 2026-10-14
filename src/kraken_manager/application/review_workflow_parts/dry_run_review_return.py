"""review_workflow: dry run review return."""
from __future__ import annotations
from kraken_manager.application.dto import (
    DryRunReviewReturnCommand,
    ReviewReturnPlan,
)
from kraken_manager.application.errors import ConflictError
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.workflows import (
    ReviewBatchState,
)

from kraken_manager.application.review_workflow_parts.review_return_inspector import _ReviewReturnInspector

class DryRunReviewReturnHandler(_ReviewReturnInspector):
    def __call__(self, command: DryRunReviewReturnCommand) -> ReviewReturnPlan:
        with self._uow_factory() as uow:
            batch = self._load_batch(
                uow,
                project_id=command.project_id,
                batch_id=command.batch_id,
                context=command.context,
                permission=Permission.RETURN_REVIEW,
            )
            self._require_revision(uow, batch, command.expected_batch_revision)
            if batch.state not in {ReviewBatchState.ISSUED, ReviewBatchState.PARTIALLY_RETURNED}:
                raise ConflictError("Only an issued review batch can receive returned files")
        prepared = self._scan_return_package(command.source)
        with self._uow_factory() as uow:
            return self._inspect(uow, batch, command.source, prepared)
