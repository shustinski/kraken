"""Local quality decisions derived from existing Karakal run records."""

from kraken_manager.application.quality_analysis import QualityAnalysisService
from kraken_manager.infrastructure.analysis import FilesystemAnalysisStore


class QualityAnalysisMixin:
    def _quality_analysis_store(self, project_id):
        store = FilesystemAnalysisStore(self.catalog_root, str(project_id))
        self._quality_events(project_id)
        return store

    def quality_analysis_runs(self, project_id):
        return QualityAnalysisService.summaries(self._quality_analysis_store(project_id), project_id)

    def quality_analysis_readiness(self, project_id, run_id):
        store = self._quality_analysis_store(project_id)
        return QualityAnalysisService.readiness(self._quality_events(project_id), store,
                                                self._projection(project_id), project_id, run_id)

    def create_quality_from_analysis(self, *, project_id, principal, run_id, rule, severity, idempotency_key):
        store = self._quality_analysis_store(project_id)
        with self._uow(str(project_id))() as unit:
            result = QualityAnalysisService.create_findings(unit, store, profiles=self.profiles,
                performers=self.performers, project_id=project_id, principal=principal, run_id=run_id,
                rule=rule, severity=severity, idempotency_key=idempotency_key)
            unit.commit()
            return result

    def recheck_quality_from_analysis(self, *, project_id, principal, issue_id, run_id,
                                       expected_revision, idempotency_key, note):
        store = self._quality_analysis_store(project_id)
        with self._uow(str(project_id))() as unit:
            result = QualityAnalysisService.recheck(unit, store, profiles=self.profiles,
                performers=self.performers, project_id=project_id, principal=principal,
                issue_id=issue_id, run_id=run_id, expected_revision=expected_revision,
                idempotency_key=idempotency_key, note=note)
            unit.commit()
            return result
