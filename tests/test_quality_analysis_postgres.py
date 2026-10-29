"""Real Karakal computation -> PostgreSQL AnalysisRun -> authenticated quality API."""

from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import pytest
import sqlalchemy as sa

from kraken_manager.domain.artifacts import ArtifactVersion, deterministic_frame_series_id
from kraken_manager.domain.identity import Performer, Principal, ProjectRole, ProjectRoleAssignment
from kraken_manager.infrastructure.filesystem._codec import decode_model
from kraken_manager.infrastructure.postgres.analysis_store import PostgresAnalysisProjectionStore
from kraken_manager.infrastructure.postgres.performer_store import PostgresPerformerStore
from kraken_manager.infrastructure.postgres.event_store import PostgresEventStore
from kraken_server.services import CommandContext, ConflictError
from tests.test_admin_project_workflows import project
from tests.test_quality_analysis import real_inputs, real_run, RULE

pytest_plugins = ["tests.test_admin_project_workflows"]


def postgres_analysis(database, root):
    services, _accounts, owner, _config = database
    created = project(services, owner)
    identifier = created["project_id"]
    layer = services.create_layer(identifier, {"name": "M1", "type": "metal", "order": 0},
                                  CommandContext(str(owner.id), "layer", created["revision"]))
    representation = services.create_representation(identifier, layer["layer_id"],
        {"name": "Mask", "kind": "image", "active": True}, CommandContext(str(owner.id), "representation", layer["revision"]))
    project_model = services.projections.get_project(identifier)
    frame = project_model.frame_id_at(1, 1)
    series = services.create_artifact_series(identifier, {"name": "1_1.png", "scope": "frame_representation",
        "layer_id": layer["layer_id"], "representation_id": representation["representation_id"], "frame_id": str(frame),
        "series_id": str(deterministic_frame_series_id(representation["representation_id"], frame))},
        CommandContext(str(owner.id), "series", None))
    initial = services.add_managed_artifact_version(identifier, series["id"],
        {"filename": "initial.png", "media_type": "image/png"}, (b"legacy placeholder never analyzed",),
        CommandContext(str(owner.id), "initial", services.artifact_stream_revision(identifier, series["id"])))
    def add_version(*, principal, project_id, series_id, source, parent_version_id, idempotency_key):
        payload = services.add_managed_artifact_version(str(project_id), str(series_id),
            {"filename": source.name, "media_type": "image/png", "parent_version_id": str(parent_version_id)},
            (source.read_bytes(),), CommandContext(str(principal.id), idempotency_key,
                                                   services.artifact_stream_revision(str(project_id), str(series_id))))
        return decode_model(ArtifactVersion, payload)
    def source_path(project_id, version_id):
        version = services.get_artifact_version(str(project_id), str(version_id))
        return services.uow_factory.blobs._path(version["sha256"])
    q = SimpleNamespace(owner=owner, project=project_model, source=decode_model(ArtifactVersion, initial),
        corrected=decode_model(ArtifactVersion, initial), service=SimpleNamespace(
            add_managed_artifact_version=add_version, artifact_source_path=source_path))
    target, reference, correction = real_inputs(q, root)
    store = PostgresAnalysisProjectionStore(services.engine)
    original, _ = real_run(q, root / "original", store, target, reference)
    checked, _ = real_run(q, root / "checked", store, correction, reference)
    services.performer_store = PostgresPerformerStore(services.engine)
    performer = services.performer_store.create(Performer.create(name="Correction engineer", color="#227788"))
    reviewer = Principal.local(subject="analysis-reviewer", display_name="Analysis reviewer")
    services.identities.save(reviewer)
    services.identities.assign(ProjectRoleAssignment.create(project_id=identifier, principal_id=reviewer.id,
        role=ProjectRole.MAINTAINER, assigned_by=owner.id))
    return services, owner, reviewer, performer, q, target, correction, original, checked


def test_real_analysis_postgres_remote_creation_recheck_readiness_and_replay(database, tmp_path):
    from fastapi.testclient import TestClient
    from kraken_hub.infrastructure.remote.client import RemoteServerProjectService
    from kraken_server.app import SessionPrincipal, create_app
    services, owner, reviewer, performer, q, target, correction, original, checked = postgres_analysis(database, tmp_path)
    project_id = str(q.project.id)
    identities = {"owner": owner, "reviewer": reviewer}
    app = create_app(services=services,
        session_resolver=lambda token: SessionPrincipal(str(identities[token].id), "local", token))
    remote = object.__new__(RemoteServerProjectService)
    active = "owner"
    with TestClient(app) as client:
        def call(method, path, *, payload=None, idempotency_key=None, if_match=None, **kwargs):
            headers = {"Authorization": f"Bearer {active}"}
            if idempotency_key is not None:
                headers["Idempotency-Key"] = idempotency_key
            if if_match is not None:
                headers["If-Match"] = str(if_match)
            response = client.request(method, path, json=payload, headers=headers)
            assert response.status_code < 400, response.text
            return response.json()
        remote._call = call
        summaries = remote.quality_analysis_runs(project_id)
        assert len(summaries) == 2 and summaries[0]["metric_keys"] == ["iou"]
        issues = remote.create_quality_from_analysis(project_id=project_id, principal=owner, run_id=original.run_id,
            rule=RULE, severity="major", idempotency_key="measured-findings")
        assert remote.create_quality_from_analysis(project_id=project_id, principal=owner, run_id=original.run_id,
            rule=RULE, severity="major", idempotency_key="measured-findings") == issues
        issue = issues[0]
        assert issue.source_version_ids == (str(target.id),)
        assert not remote.quality_analysis_readiness(project_id, original.run_id)["ready"]
        issue = remote.mutate_quality_issue(project_id=project_id, principal=owner, issue_id=issue.id,
            expected_revision=issue.revision, operation="assign", arguments={"assignee_id": str(performer.id),
                "note": "Repair measured mismatch"}, idempotency_key="assign")
        issue = remote.mutate_quality_issue(project_id=project_id, principal=owner, issue_id=issue.id,
            expected_revision=issue.revision, operation="correct", arguments={"evidence_version_ids": [str(correction.id)],
                "note": "Repair version submitted"}, idempotency_key="correct")
        base = f"/api/v1/projects/{project_id}/quality-issues/{issue.id}"
        response = client.post(base + "/commands", json={"operation": "recheck", "arguments": {
            "passed": True, "evidence_version_ids": [str(correction.id)], "note": "Spoofed analysis verdict"}},
            headers={"Authorization": "Bearer reviewer", "Idempotency-Key": "spoof", "If-Match": str(issue.revision)})
        assert response.status_code == 422, response.text
        active = "reviewer"
        issue = remote.recheck_quality_from_analysis(project_id=project_id, principal=reviewer, issue_id=issue.id,
            run_id=checked.run_id, expected_revision=issue.revision, idempotency_key="measured-recheck",
            note="Independent measured verification")
        assert issue.reviews[-1].observations == ((str(q.project.frame_id_at(1, 1)), 1.0),)
        assert issue.reviews[-1].passed
        issue = remote.mutate_quality_issue(project_id=project_id, principal=reviewer, issue_id=issue.id,
            expected_revision=issue.revision, operation="accept", arguments={"note": "Accept measured correction"},
            idempotency_key="accept")
        assert issue.state.value == "accepted"
        assert remote.quality_analysis_readiness(project_id, checked.run_id)["ready"]


def test_postgres_analysis_evaluation_receipt_cas_and_atomic_rollback(database, tmp_path, monkeypatch):
    services, owner, _reviewer, _performer, q, _target, _correction, original, _checked = postgres_analysis(database, tmp_path)
    project_id = str(q.project.id)
    append = PostgresEventStore.append
    def fail_receipt(self, stream_id, *, expected_revision, events):
        result = append(self, stream_id, expected_revision=expected_revision, events=events)
        if stream_id.startswith("quality-evaluation:"):
            raise OSError("injected evaluation receipt failure")
        return result
    monkeypatch.setattr(PostgresEventStore, "append", fail_receipt)
    payload = {"run_id": original.run_id, "rule": RULE, "severity": "major"}
    with pytest.raises(OSError, match="receipt failure"):
        services.transport_quality_analysis(project_id, payload, CommandContext(str(owner.id), "atomic-evaluation", None))
    assert services.list_quality_issues(project_id) == []
    with services.engine.connect() as connection:
        assert connection.execute(sa.text("SELECT count(*) FROM transactional_outbox WHERE event_type LIKE 'quality.%'")) .scalar_one() == 0
    barrier = Barrier(2)
    def synchronized(self, stream_id, *, expected_revision, events):
        if stream_id.startswith("quality-evaluation:"):
            barrier.wait(timeout=10)
        return append(self, stream_id, expected_revision=expected_revision, events=events)
    monkeypatch.setattr(PostgresEventStore, "append", synchronized)
    payload = {"run_id": original.run_id, "rule": RULE | {"comparison": "gt", "threshold": 1.0}, "severity": "major"}
    context = CommandContext(str(owner.id), "concurrent-zero-result", None)
    def evaluate(_index):
        try:
            return services.transport_quality_analysis(project_id, payload, context)
        except ConflictError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(evaluate, range(2)))
    assert results.count({"items": []}) == 1 and results.count(None) == 1
    monkeypatch.setattr(PostgresEventStore, "append", append)
    assert services.transport_quality_analysis(project_id, payload, context) == {"items": []}
    assert services.list_quality_issues(project_id) == []
