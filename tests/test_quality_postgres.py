"""Quality commands against PostgreSQL in fixture-owned UUID schemas."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import sqlalchemy as sa

from kraken_manager.domain.identity import Performer, Principal, ProjectRole, ProjectRoleAssignment
from kraken_manager.infrastructure.postgres.event_store import PostgresEventStore
from kraken_manager.infrastructure.postgres.performer_store import PostgresPerformerStore
from kraken_server.services import CommandContext, ConflictError
from tests.test_admin_project_workflows import project

pytest_plugins = ["tests.test_admin_project_workflows"]


def setup_quality(database):
    services, _accounts, owner, _config = database
    created = project(services, owner)
    identifier = created["project_id"]
    layer = services.create_layer(identifier, {"name": "M1", "type": "metal", "order": 0},
                                  CommandContext(str(owner.id), "layer", created["revision"]))
    services.performer_store = PostgresPerformerStore(services.engine)
    performer = services.performer_store.create(Performer.create(name="Repair engineer", color="#225588"))
    series = services.create_artifact_series(identifier, {"name": "Measurement", "scope": "layer_attachment",
        "layer_id": layer["layer_id"]}, CommandContext(str(owner.id), "series", None))
    source = services.add_managed_artifact_version(identifier, series["id"],
        {"filename": "measurement.txt", "media_type": "text/plain"}, (b"gap detected",),
        CommandContext(str(owner.id), "source-version", services.artifact_stream_revision(identifier, series["id"])))
    corrected = services.add_managed_artifact_version(identifier, series["id"],
        {"filename": "measurement.txt", "media_type": "text/plain", "parent_version_id": source["id"]},
        (b"gap corrected",), CommandContext(str(owner.id), "corrected-version",
                                             services.artifact_stream_revision(identifier, series["id"])))
    args = {"title": "Conductor gap", "description": "Interrupted conductor", "severity": "major",
            "scope": {"layer_ids": [layer["layer_id"]]}, "source_version_ids": [source["id"]]}
    return services, owner, identifier, performer, source, corrected, args


def test_postgres_quality_create_replay_reverse_lookup_and_concurrent_cas(database, monkeypatch):
    services, owner, identifier, _performer, source, _corrected, args = setup_quality(database)
    context = CommandContext(str(owner.id), "quality", None)
    issue = services.transport_create_quality_issue(identifier, args, context)
    assert services.transport_create_quality_issue(identifier, args, context) == issue
    with pytest.raises(ConflictError):
        services.transport_create_quality_issue(identifier, args | {"title": "Different request"}, context)
    barrier = Barrier(2)
    append = PostgresEventStore.append
    def synchronized(self, stream_id, *, expected_revision, events):
        if stream_id == f"quality-issue:{issue['id']}" and expected_revision == 1:
            barrier.wait(timeout=10)
        return append(self, stream_id, expected_revision=expected_revision, events=events)
    monkeypatch.setattr(PostgresEventStore, "append", synchronized)
    def write(index):
        try:
            return services.transport_mutate_quality_issue(identifier, issue["id"], {
                "operation": "exception", "arguments": {"note": f"Authorized exception {index}"},
            }, CommandContext(str(owner.id), f"exception-{index}", 0))
        except ConflictError:
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, range(2)))
    assert sum(result is not None for result in results) == 1
    persisted = services.get_quality_issue(identifier, issue["id"])
    assert persisted["revision"] == 1 and persisted["state"] == "exception"
    assert services.list_quality_issues(identifier, artifact_version_id=source["id"]) == [persisted]
    assert len(services.quality_issue_history(identifier, issue["id"])) == 2
    with services.engine.connect() as connection:
        assert connection.execute(sa.text("SELECT count(*) FROM transactional_outbox WHERE event_type LIKE 'quality.%'")) .scalar_one() == 2


def test_postgres_quality_write_failure_rolls_back_event_outbox_and_command_key(database, monkeypatch):
    services, owner, identifier, _performer, _source, _corrected, args = setup_quality(database)
    append = PostgresEventStore.append
    def fail_after_append(self, stream_id, *, expected_revision, events):
        result = append(self, stream_id, expected_revision=expected_revision, events=events)
        if stream_id.startswith("quality-issue:"):
            raise OSError("injected quality persistence failure")
        return result
    monkeypatch.setattr(PostgresEventStore, "append", fail_after_append)
    context = CommandContext(str(owner.id), "quality-rollback", None)
    with pytest.raises(OSError, match="injected"):
        services.transport_create_quality_issue(identifier, args, context)
    assert services.list_quality_issues(identifier) == []
    with services.engine.connect() as connection:
        assert connection.execute(sa.text("SELECT count(*) FROM transactional_outbox WHERE event_type LIKE 'quality.%'")) .scalar_one() == 0
        assert connection.execute(sa.text("SELECT count(*) FROM command_idempotency WHERE idempotency_key = 'quality-rollback'")) .scalar_one() == 0
    monkeypatch.setattr(PostgresEventStore, "append", append)
    assert services.transport_create_quality_issue(identifier, args, context)["revision"] == 0


def test_postgres_quality_full_http_correction_recheck_accept_and_rbac(database):
    from fastapi.testclient import TestClient
    from kraken_server.app import SessionPrincipal, create_app
    services, owner, identifier, performer, source, corrected, args = setup_quality(database)
    reviewer = Principal.local(subject="quality-reviewer", display_name="Quality reviewer")
    outsider = Principal.local(subject="outsider", display_name="Outside project")
    for person in (reviewer, outsider):
        services.identities.save(person)
    services.identities.assign(ProjectRoleAssignment.create(project_id=identifier, principal_id=reviewer.id,
        role=ProjectRole.MAINTAINER, assigned_by=owner.id))
    people = {"owner": owner, "reviewer": reviewer, "outsider": outsider}
    app = create_app(services=services,
        session_resolver=lambda token: SessionPrincipal(str(people[token].id), "local", token))
    with TestClient(app) as client:
        base = f"/api/v1/projects/{identifier}/quality-issues"
        assert client.get(base).status_code == 401
        response = client.post(base, json=args, headers={"Authorization": "Bearer owner", "Idempotency-Key": "http-issue"})
        assert response.status_code == 201, response.text
        issue = response.json()
        def command(operation, arguments, actor="owner"):
            return client.post(f"{base}/{issue['id']}/commands", json={"operation": operation, "arguments": arguments},
                headers={"Authorization": f"Bearer {actor}", "Idempotency-Key": f"http-{operation}-{actor}",
                         "If-Match": str(issue["revision"])})
        assert client.get(base, headers={"Authorization": "Bearer outsider"}).status_code == 403
        assert command("exception", {"note": "Unauthorized waiver"}, "outsider").status_code == 403
        response = command("assign", {"assignee_id": str(performer.id), "note": "Repair conductor"})
        assert response.status_code == 200, response.text
        issue = response.json()
        response = command("correct", {"evidence_version_ids": [corrected["id"]], "note": "Correction complete"})
        assert response.status_code == 200, response.text
        issue = response.json()
        assert command("recheck", {"evidence_version_ids": [corrected["id"]], "passed": True, "note": "Self check"}).status_code == 403
        response = command("recheck", {"evidence_version_ids": [corrected["id"]], "passed": True,
            "note": "Continuity verified"}, "reviewer")
        assert response.status_code == 200, response.text
        issue = response.json()
        response = command("accept", {"note": "Accepted based on continuity check"}, "reviewer")
        assert response.status_code == 200, response.text
        issue = response.json()
        assert issue["state"] == "accepted" and issue["revision"] == 4
        response = client.get(base, params={"artifact_version_id": source["id"]}, headers={"Authorization": "Bearer owner"})
        assert response.json()["items"] == [issue]
        assert len(services.quality_issue_history(identifier, issue["id"])) == 5
