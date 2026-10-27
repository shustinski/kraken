"""Backend-neutral quality reads and reverse artifact lookup."""

from kraken_manager.application.quality import QualityService
from kraken_manager.domain.common import validate_uuid
from kraken_manager.domain.quality import QualityIssue


def list_quality_issues(events, project_id, *, artifact_version_id=None, frame_id=None):
    project_id = validate_uuid(str(project_id))
    if artifact_version_id is not None:
        artifact_version_id = validate_uuid(artifact_version_id)
    if frame_id is not None:
        frame_id = validate_uuid(frame_id)
    issues = []
    for issue in QualityService.list_issues(events, project_id):
        if issue.project_id != project_id or (frame_id is not None and frame_id not in issue.scope.frame_ids):
            continue
        versions = set(issue.source_version_ids) | set(issue.correction_version_ids)
        versions.update(v for review in issue.reviews for v in review.evidence_version_ids)
        if artifact_version_id is not None and artifact_version_id not in versions:
            continue
        issues.append(issue)
    return tuple(sorted(issues, key=lambda i: (i.updated_at, i.id), reverse=True))


def quality_history(events, project_id, issue_id):
    QualityService.load(events, project_id, issue_id)
    return tuple({
        "event_id": event.event_id, "revision": event.revision, "operation": event.event_type,
        "actor_id": str(event.actor.principal_id), "at": event.recorded_at.isoformat(),
        "issue": QualityIssue.from_dict(event.payload["issue"]).to_dict(),
    } for event in events.load_stream(QualityService.stream_id(issue_id)))
