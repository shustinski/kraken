"""HTTP access to stored analysis quality evaluations."""

from kraken_manager.domain.quality import QualityIssue


class QualityAnalysisMixin:
    def quality_analysis_runs(self, project_id):
        return tuple(self._call("GET", f"/api/v1/projects/{project_id}/quality-analysis-runs")["items"])

    def quality_analysis_readiness(self, project_id, run_id):
        from urllib.parse import quote
        return self._call("GET", f"/api/v1/projects/{project_id}/quality-analysis-runs/{quote(run_id, safe='')}/readiness")

    def create_quality_from_analysis(self, *, project_id, principal, run_id, rule, severity, idempotency_key):
        result = self._call("POST", f"/api/v1/projects/{project_id}/quality-issues/from-analysis",
            payload={"run_id": run_id, "rule": rule, "severity": severity}, idempotency_key=idempotency_key)
        return tuple(QualityIssue.from_dict(item) for item in result["items"])

    def recheck_quality_from_analysis(self, *, project_id, principal, issue_id, run_id,
                                       expected_revision, idempotency_key, note):
        return QualityIssue.from_dict(self._call("POST",
            f"/api/v1/projects/{project_id}/quality-issues/{issue_id}/analysis-recheck",
            payload={"run_id": run_id, "note": note}, idempotency_key=idempotency_key, if_match=expected_revision))
