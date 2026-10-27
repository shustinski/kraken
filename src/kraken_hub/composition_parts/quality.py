"""Local quality findings reuse project events and artifact/review ownership."""

from kraken_manager.application.errors import NotFoundError
from kraken_manager.application.quality import QualityService
from kraken_manager.infrastructure.filesystem import FilesystemEventStore
from kraken_manager.infrastructure.quality_events import list_quality_issues, quality_history
from .quality_analysis import QualityAnalysisMixin


class QualityMixin(QualityAnalysisMixin):
    def quality_frame_versions(self, project_id, layer_id, x, y):
        return QualityService.frame_versions(self._projection(project_id), project_id, layer_id, x, y)

    def create_quality_issue(self, *, project_id, principal, arguments, idempotency_key):
        return self.mutate_quality_issue(
            project_id=project_id, principal=principal, issue_id=None, expected_revision=None,
            operation="create", arguments=arguments, idempotency_key=idempotency_key,
        )

    def mutate_quality_issue(self, *, project_id, principal, issue_id, expected_revision,
                             operation, arguments, idempotency_key):
        with self._uow(str(project_id))() as unit:
            result = QualityService.execute(
                unit, profiles=self.profiles, performers=self.performers, project_id=project_id,
                principal=principal, issue_id=issue_id, expected_revision=expected_revision,
                operation=operation, arguments=arguments, idempotency_key=idempotency_key,
            )
            unit.commit()
            return result

    def _quality_events(self, project_id):
        if self._projection(project_id).get_project(project_id) is None:
            raise NotFoundError("Project was not found")
        return FilesystemEventStore(self.catalog_root, str(project_id))

    def get_quality_issue(self, project_id, issue_id):
        return QualityService.load(self._quality_events(project_id), project_id, issue_id)

    def list_quality_issues(self, project_id, *, artifact_version_id=None, frame_id=None):
        return list_quality_issues(self._quality_events(project_id), project_id,
                                   artifact_version_id=artifact_version_id, frame_id=frame_id)

    def quality_issue_history(self, project_id, issue_id):
        return quality_history(self._quality_events(project_id), project_id, issue_id)
