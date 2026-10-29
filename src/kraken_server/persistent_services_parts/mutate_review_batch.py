"""persistent_services: mutate review batch."""
from __future__ import annotations
from kraken_server.project_deletion import guarded_storage
from collections.abc import Iterator, Mapping
from typing import Any
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    AcceptReviewCommand,
    CancelReviewBatchCommand,
    CommitReviewReturnCommand,
    DryRunReviewReturnCommand,
    ExportReviewPackageCommand,
    RequestReviewChangesCommand,
)
from kraken_manager.application.errors import (
    NotFoundError as ApplicationNotFoundError,
)
from kraken_manager.domain.common import (
    ArtifactVersionId,
    ReviewBatchId,
    validate_uuid,
)
from kraken_manager.domain.events import EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.infrastructure.filesystem._codec import encode_model
from kraken_server.services import CommandContext, ConflictError, NotFoundError, ValidationError

from kraken_server.persistent_services_parts.layer_source_scan import _project_id


class MutateReviewBatchMixin:
    def mutate_review_batch(
        self,
        project_id: str,
        batch_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        common = {
            "context": self._application_context(context),
            "project_id": _project_id(project_id),
            "batch_id": ReviewBatchId(validate_uuid(batch_id, field="batch_id")),
            "expected_batch_revision": self._require_revision(context),
        }
        try:
            action = str(payload.get("action", ""))
            if action == "accept":
                raw_ids = payload.get("candidate_version_ids", ())
                if not isinstance(raw_ids, list):
                    raise ValueError("candidate_version_ids must be an array")
                value = self._accept_review(
                    AcceptReviewCommand(
                        candidate_version_ids=tuple(
                            ArtifactVersionId(validate_uuid(str(item), field="version_id"))
                            for item in raw_ids
                        ),
                        **common,
                    )
                )
            elif action == "request_changes":
                value = self._request_review_changes(
                    RequestReviewChangesCommand(reason=str(payload.get("reason", "")), **common)
                )
            elif action == "cancel":
                value = self._cancel_review_batch(CancelReviewBatchCommand(**common))
            else:
                raise ValidationError("Unknown review operation")
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    @guarded_storage
    def export_review_package(
        self,
        project_id: str,
        batch_id: str,
        destination: str,
        context: CommandContext,
    ) -> dict[str, Any]:
        if self._export_review_package is None:
            raise ConflictError("Server review-package signing is not configured")
        try:
            value = self._export_review_package(
                ExportReviewPackageCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    batch_id=ReviewBatchId(
                        validate_uuid(batch_id, field="review_batch_id")
                    ),
                    expected_batch_revision=self._require_revision(context),
                    destination=destination,
                    include_images=True,
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def inspect_review_return(
        self,
        project_id: str,
        batch_id: str,
        source: str,
        context: CommandContext,
    ) -> dict[str, Any]:
        if self._dry_run_review_return is None:
            raise ConflictError("Server review-package verification is not configured")
        try:
            value = self._dry_run_review_return(
                DryRunReviewReturnCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    batch_id=ReviewBatchId(
                        validate_uuid(batch_id, field="review_batch_id")
                    ),
                    expected_batch_revision=self._require_revision(context),
                    source=source,
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def commit_review_return(
        self,
        project_id: str,
        batch_id: str,
        source: str,
        context: CommandContext,
    ) -> dict[str, Any]:
        if self._commit_review_return is None:
            raise ConflictError("Server review-package verification is not configured")
        try:
            value = self._commit_review_return(
                CommitReviewReturnCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    batch_id=ReviewBatchId(
                        validate_uuid(batch_id, field="review_batch_id")
                    ),
                    expected_batch_revision=self._require_revision(context),
                    source=source,
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def list_plugin_jobs(self, project_id: str | None = None) -> list[dict[str, Any]]:
        projects = self.projections.list_projects(include_archived=True)
        if project_id is not None:
            projects = tuple(item for item in projects if str(item.id) == project_id)
            if not projects:
                raise NotFoundError("Project was not found")
        removed = {
            str(event.payload.get("plugin_job_id", ""))
            for project in projects
            for event in self._pipeline_events(str(project.id))
            if event.event_type == "LayerPipelineActionRemoved" and event.payload.get("remove_task")
        }
        jobs = [
            job
            for project in projects
            for job in self.projections.list_plugin_jobs(project.id)
            if str(job.id) not in removed
        ]
        jobs.sort(key=lambda item: (item.updated_at, str(item.id)), reverse=True)
        return [encode_model(item) for item in jobs]

    def _pipeline_events(self, project_id: str) -> Iterator[EventEnvelope]:
        after = 0
        while True:
            page = self.events.list_project_events(_project_id(project_id), after_position=after, limit=500)
            if not page:
                return
            for _position, event in page:
                yield event
            after = page[-1][0]

    def authorize_desktop_job(self, project_id: str, principal_id: str) -> None:
        actor = self._actor(principal_id)
        try:
            with self.uow_factory() as uow:
                project = uow.projections.get_project(_project_id(project_id))
                if project is None:
                    raise ApplicationNotFoundError("Project was not found")
                profile = self.profiles.get(project.storage_profile)
                if profile is None:
                    raise ApplicationNotFoundError("Storage profile was not found")
                AuthorizationPolicy().require(
                    principal=actor,
                    storage=profile,
                    permission=Permission.RUN_PLUGIN,
                    roles=uow.acl.roles_for(project.id, actor.id),
                    gitlab_identity_verified=True,
                )
        except Exception as exc:
            self._translate_lifecycle_error(exc)
