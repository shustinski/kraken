"""Remote quality issue operations share the local service signatures."""

from urllib.parse import urlencode

from kraken_manager.domain.quality import QualityIssue
from .quality_analysis import QualityAnalysisMixin


class QualityMixin(QualityAnalysisMixin):
    def quality_frame_versions(self, project_id, layer_id, x, y):
        from kraken_manager.domain.artifacts import ArtifactVersion
        from kraken_manager.infrastructure.filesystem._codec import decode_model
        query = urlencode({"layer_id": layer_id, "x": x, "y": y})
        payload = self._call("GET", f"/api/v1/projects/{project_id}/quality-artifacts?{query}")
        return tuple(decode_model(ArtifactVersion, item) for item in payload["items"])

    def list_quality_issues(self, project_id, *, artifact_version_id=None, frame_id=None):
        filters = {key: value for key, value in {
            "artifact_version_id": artifact_version_id, "frame_id": frame_id,
        }.items() if value is not None}
        suffix = "?" + urlencode(filters) if filters else ""
        payload = self._call("GET", f"/api/v1/projects/{project_id}/quality-issues{suffix}")
        return tuple(QualityIssue.from_dict(item) for item in payload["items"])

    def get_quality_issue(self, project_id, issue_id):
        return QualityIssue.from_dict(self._call("GET", f"/api/v1/projects/{project_id}/quality-issues/{issue_id}"))

    def quality_issue_history(self, project_id, issue_id):
        return tuple(self._call("GET", f"/api/v1/projects/{project_id}/quality-issues/{issue_id}/history")["items"])

    def create_quality_issue(self, *, project_id, principal, arguments, idempotency_key):
        return QualityIssue.from_dict(self._call("POST", f"/api/v1/projects/{project_id}/quality-issues",
            payload=arguments, idempotency_key=idempotency_key))

    def mutate_quality_issue(self, *, project_id, principal, issue_id, expected_revision,
                             operation, arguments, idempotency_key):
        return QualityIssue.from_dict(self._call("POST", f"/api/v1/projects/{project_id}/quality-issues/{issue_id}/commands",
            payload={"operation": operation, "arguments": arguments},
            idempotency_key=idempotency_key, if_match=expected_revision))
