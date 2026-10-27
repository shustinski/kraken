"""Server quality decisions over transaction-bound PostgreSQL AnalysisRun data."""

from kraken_manager.application.quality_analysis import QualityAnalysisService
from kraken_manager.infrastructure.postgres.analysis_store import PostgresAnalysisProjectionStore


class QualityAnalysisMixin:
    def quality_analysis_runs(self, project_id):
        try:
            self._quality_project(project_id)
            return QualityAnalysisService.summaries(PostgresAnalysisProjectionStore(self.engine), project_id)
        except Exception as exc:
            self._quality_error(exc)

    def quality_analysis_readiness(self, project_id, run_id):
        try:
            self._quality_project(project_id)
            return QualityAnalysisService.readiness(self.events, PostgresAnalysisProjectionStore(self.engine),
                                                    self.projections, project_id, run_id)
        except Exception as exc:
            self._quality_error(exc)

    def transport_quality_analysis(self, project_id, payload, context, *, issue_id=None):
        try:
            allowed = {"run_id", "rule", "severity"} if issue_id is None else {"run_id", "note"}
            if set(payload) != allowed:
                raise ValueError("Invalid analysis quality command fields")
            with self.uow_factory() as unit:
                store = PostgresAnalysisProjectionStore(self.engine, connection=unit._connection)
                arguments = dict(unit=unit, store=store, profiles=self.profiles, performers=self.performer_store,
                    project_id=project_id, principal=self._actor(context.actor_id), idempotency_key=context.idempotency_key,
                    gitlab_identity_verified=True, **payload)
                if issue_id is None:
                    result = QualityAnalysisService.create_findings(**arguments)
                    response = {"items": [issue.to_dict() for issue in result]}
                else:
                    result = QualityAnalysisService.recheck(**arguments, issue_id=issue_id,
                                                            expected_revision=context.expected_revision)
                    response = result.to_dict()
                unit.commit()
                return response
        except Exception as exc:
            self._quality_error(exc)
