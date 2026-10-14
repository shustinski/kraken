"""review_workflow: plan review package handler."""
from __future__ import annotations
from kraken_manager.application.dto import (
    PlanReviewPackageCommand,
    ReviewPackagePlan,
)
from kraken_manager.application.errors import ConflictError
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.workflows import (
    ReviewBatchState,
)

from kraken_manager.application.review_workflow_parts.review_package_planner import _ReviewPackagePlanner

class PlanReviewPackageHandler(_ReviewPackagePlanner):
    def __call__(self, command: PlanReviewPackageCommand) -> ReviewPackagePlan:
        with self._uow_factory() as uow:
            batch = self._load_batch(
                uow,
                project_id=command.project_id,
                batch_id=command.batch_id,
                context=command.context,
                permission=Permission.MANAGE_REVIEW,
            )
            self._require_revision(uow, batch, command.expected_batch_revision)
            if batch.state not in {ReviewBatchState.DRAFT, ReviewBatchState.CHANGES_REQUESTED}:
                raise ConflictError("Only a draft or changes-requested batch can be exported")
            plan, _ = self._build_plan(
                uow,
                batch,
                package_id=command.package_id,
                issued_by=command.context.actor.id,
                include_images=command.include_images,
                issued_at=self._clock.now(),
            )
            return plan
