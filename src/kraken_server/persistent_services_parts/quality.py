"""Authenticated transport for quality findings in the project event store."""

from kraken_manager.application.errors import NotFoundError
from kraken_manager.application.quality import QualityService
from kraken_manager.infrastructure.quality_events import list_quality_issues, quality_history
from .quality_analysis import QualityAnalysisMixin


class QualityMixin(QualityAnalysisMixin):
    def quality_frame_versions(self, project_id, layer_id, x, y):
        from kraken_manager.infrastructure.filesystem._codec import encode_model
        try:
            return [encode_model(v) for v in QualityService.frame_versions(self.projections, project_id, layer_id, x, y)]
        except Exception as exc:
            self._quality_error(exc)

    def transport_create_quality_issue(self, project_id, payload, context):
        return self._quality_command(project_id, None, "create", payload, context)

    def transport_mutate_quality_issue(self, project_id, issue_id, payload, context):
        try:
            if set(payload) != {"operation", "arguments"}:
                raise ValueError("Quality commands require operation and arguments")
            return self._quality_command(project_id, issue_id, payload["operation"], payload["arguments"], context)
        except Exception as exc:
            self._quality_error(exc)

    def _quality_command(self, project_id, issue_id, operation, arguments, context):
        try:
            if operation == "create" and issue_id is not None:
                raise ValueError("Create a new quality issue at the collection endpoint")
            with self.uow_factory() as unit:
                issue = QualityService.execute(
                    unit, profiles=self.profiles, performers=self.performer_store,
                    project_id=project_id, principal=self._actor(context.actor_id), issue_id=issue_id,
                    operation=operation, arguments=arguments, expected_revision=context.expected_revision,
                    idempotency_key=context.idempotency_key, gitlab_identity_verified=True,
                )
                unit.commit()
                return issue.to_dict()
        except Exception as exc:
            self._quality_error(exc)

    def _quality_project(self, project_id):
        from kraken_manager.domain.common import validate_uuid
        if self.projections.get_project(validate_uuid(str(project_id))) is None:
            raise NotFoundError("Project was not found")

    def list_quality_issues(self, project_id, *, artifact_version_id=None, frame_id=None):
        try:
            self._quality_project(project_id)
            return [issue.to_dict() for issue in list_quality_issues(
                self.events, project_id, artifact_version_id=artifact_version_id, frame_id=frame_id,
            )]
        except Exception as exc:
            self._quality_error(exc)

    def get_quality_issue(self, project_id, issue_id):
        try:
            self._quality_project(project_id)
            return QualityService.load(self.events, project_id, issue_id).to_dict()
        except Exception as exc:
            self._quality_error(exc)

    def quality_issue_history(self, project_id, issue_id):
        try:
            self._quality_project(project_id)
            return quality_history(self.events, project_id, issue_id)
        except Exception as exc:
            self._quality_error(exc)

    def _quality_error(self, exc):
        from kraken_manager.domain.common import InvalidStateTransition
        from kraken_manager.infrastructure.filesystem.event_store import EventStreamConflict
        from kraken_server.services import ConflictError, ValidationError
        if isinstance(exc, (InvalidStateTransition, EventStreamConflict)):
            raise ConflictError(str(exc)) from exc
        if isinstance(exc, KeyError) and not isinstance(exc, NotFoundError):
            raise ValidationError(f"Required quality field is missing: {exc}") from exc
        self._translate_lifecycle_error(exc)
        raise exc
