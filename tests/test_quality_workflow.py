"""Quality lifecycle through real local transactions and server transport."""

from dataclasses import replace
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import threading
from uuid import uuid4

import pytest

from kraken_manager.application.errors import AuthorizationError, ConcurrencyError, ConflictError
from kraken_manager.domain.common import DomainValidationError, InvalidStateTransition
from kraken_manager.domain.identity import Principal, ProjectRole
from kraken_manager.domain.quality import QualityState
from kraken_manager.infrastructure.filesystem import FilesystemEventStore
from kraken_server.persistent_services import PostgresServerServices
from kraken_server.services import CommandContext
from tests.test_work_orders import _project_with_frames, _service


@pytest.fixture
def quality(tmp_path, monkeypatch):
    monkeypatch.setenv("KRAKEN_PROJECT_ACCESS_MODE", "acl")
    service = _service(tmp_path)
    owner, performer, project, layer, images, _ = _project_with_frames(service, tmp_path / "source", with_vectors=False)
    source_series = next(s for s in service.list_artifact_series(project.id)
                         if str(s.frame_id) == str(project.frame_id_at(1, 1)))
    source = service.artifact_versions(project.id, source_series.id)[0]
    changed_file = tmp_path / "corrected.png"
    changed_file.write_bytes(b"corrected image")
    corrected = service.add_managed_artifact_version(
        principal=owner.principal, project_id=project.id, series_id=source_series.id,
        source=changed_file, parent_version_id=source.id, idempotency_key="corrected-version",
    )
    reviewer = Principal.local(subject="quality-reviewer", display_name="Independent reviewer")
    service.identities.save(reviewer)
    service.assign_project_role(
        principal=owner.principal, project=project, target_principal_id=reviewer.id,
        role=ProjectRole.MAINTAINER, expected_revision=0, idempotency_key="reviewer-acl",
    )
    arguments = {"title": "Open conductor", "description": "A gap interrupts the trace",
                 "severity": "major", "source_version_ids": [str(source.id)],
                 "scope": {"layer_ids": [str(layer.id)], "frame_coordinates": [[1, 1]]}}

    def create(**overrides):
        return service.create_quality_issue(project_id=project.id, principal=owner.principal,
                                             arguments=arguments | overrides, idempotency_key="quality")

    def command(issue, operation, args, *, principal=None, key=None, revision=None):
        return service.mutate_quality_issue(
            project_id=project.id, issue_id=issue.id, principal=principal or owner.principal,
            expected_revision=issue.revision if revision is None else revision,
            operation=operation, arguments=args, idempotency_key=key or f"{operation}:{issue.revision}",
        )

    return SimpleNamespace(service=service, owner=owner.principal, reviewer=reviewer, performer=performer,
                           project=project, layer=layer, source=source, corrected=corrected,
                           create=create, command=command, arguments=arguments)


def corrected_issue(q):
    issue = q.create()
    issue = q.command(issue, "assign", {"assignee_id": str(q.performer.id), "note": "Repair the gap"})
    return q.command(issue, "correct", {"evidence_version_ids": [str(q.corrected.id)], "note": "Gap corrected"})


def test_local_issue_correction_recheck_accept_history_and_reverse_lookup(quality):
    q = quality
    issue = corrected_issue(q)
    with pytest.raises(AuthorizationError, match="own correction"):
        q.command(issue, "recheck", {"passed": True, "evidence_version_ids": [str(q.corrected.id)], "note": "Self review"})
    checked = q.command(issue, "recheck", {"passed": True, "evidence_version_ids": [str(q.corrected.id)],
                                            "note": "Connectivity verified"}, principal=q.reviewer)
    accepted = q.command(checked, "accept", {"note": "Conductor continuity restored"}, principal=q.reviewer)
    assert accepted.state is QualityState.ACCEPTED
    assert accepted.source_version_ids == (str(q.source.id),)
    assert accepted.correction_version_ids == (str(q.corrected.id),)
    assert q.service.get_quality_issue(q.project.id, accepted.id) == accepted
    assert q.service.list_quality_issues(q.project.id, artifact_version_id=str(q.source.id)) == (accepted,)
    assert q.service.list_quality_issues(q.project.id, artifact_version_id=str(q.corrected.id)) == (accepted,)
    assert q.service.list_quality_issues(q.project.id, frame_id=str(q.project.frame_id_at(2, 1))) == ()
    history = q.service.quality_issue_history(q.project.id, accepted.id)
    assert [e["operation"] for e in history] == ["quality.create", "quality.assign", "quality.correct",
                                                 "quality.recheck", "quality.accept"]
    assert history[-1]["actor_id"] == str(q.reviewer.id)
    assert [e["revision"] for e in history] == [1, 2, 3, 4, 5]
    # Quality data is authoritative in the same event stream, independent of SQLite caches.
    q.service._projection(q.project.id).destroy_cache()
    assert q.service.get_quality_issue(q.project.id, accepted.id) == accepted
    with pytest.raises(InvalidStateTransition):
        q.command(accepted, "exception", {"note": "Cannot revise a closed acceptance"})


def test_failed_recheck_reject_reassign_and_document_exception(quality):
    q = quality
    corrected = corrected_issue(q)
    checked = q.command(corrected, "recheck", {"passed": False, "evidence_version_ids": [str(q.corrected.id)],
                                                "note": "Gap remains"}, principal=q.reviewer)
    with pytest.raises(InvalidStateTransition, match="failed recheck"):
        q.command(checked, "accept", {"note": "Invalid acceptance"}, principal=q.reviewer)
    rejected = q.command(checked, "reject", {"note": "Repeat repair"}, principal=q.reviewer)
    assigned = q.command(rejected, "assign", {"assignee_id": str(q.performer.id), "note": "Second pass"})
    exception = q.command(assigned, "exception", {"note": "Documented design discontinuity; exception approved"},
                          principal=q.reviewer)
    assert exception.state is QualityState.EXCEPTION
    assert [r.action for r in exception.reviews] == ["assign", "correct", "recheck", "reject", "assign", "exception"]


def test_quality_replay_cas_bad_scope_and_foreign_evidence_do_not_append(quality):
    q = quality
    issue = q.create()
    assert q.create() == issue
    with pytest.raises(ConflictError):
        q.create(title="Changed request with the same key")
    assigned = q.command(issue, "assign", {"assignee_id": str(q.performer.id), "note": "Repair"}, key="assign")
    assert q.command(issue, "assign", {"assignee_id": str(q.performer.id), "note": "Repair"}, key="assign") == assigned
    count = len(q.service.quality_issue_history(q.project.id, issue.id))
    for evidence in ([str(uuid4())], [str(q.source.id)]):
        with pytest.raises(DomainValidationError):
            q.command(assigned, "correct", {"evidence_version_ids": evidence, "note": "Untrusted correction"})
    with pytest.raises(ConcurrencyError):
        q.command(assigned, "exception", {"note": "Stale"}, revision=0)
    with pytest.raises(DomainValidationError):
        q.service.create_quality_issue(project_id=q.project.id, principal=q.owner,
            arguments=q.arguments | {"scope": {"layer_ids": [str(q.layer.id)], "frame_coordinates": [[2, 1]]}},
            idempotency_key="wrong-frame")
    assert len(q.service.quality_issue_history(q.project.id, issue.id)) == count


@pytest.mark.parametrize("operation,args", [
    ("assign", {"assignee_id": str(uuid4()), "note": "Unauthorized"}),
    ("exception", {"note": "Unauthorized waiver"}),
    ("accept", {"note": "Unauthorized acceptance"}),
])
def test_quality_mutations_require_backend_permission(quality, operation, args):
    q = quality
    issue = q.create()
    outsider = Principal.local(subject="outsider", display_name="Outsider")
    q.service.identities.save(outsider)
    with pytest.raises(AuthorizationError):
        q.command(issue, operation, args, principal=outsider)
    assert len(q.service.quality_issue_history(q.project.id, issue.id)) == 1


def test_two_quality_writers_cannot_lose_a_revision(quality):
    q = quality
    issue = q.create()
    barrier = threading.Barrier(2)
    def write(index):
        barrier.wait(timeout=5)
        try:
            return q.command(issue, "exception", {"note": f"Reviewed disposition {index}"}, key=f"writer-{index}")
        except ConcurrencyError:
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, range(2)))
    assert len([result for result in results if result is not None]) == 1
    assert len(q.service.quality_issue_history(q.project.id, issue.id)) == 2


def test_corrector_cannot_record_another_linked_assignees_correction(quality):
    q = quality
    issue = q.create()
    q.service.performers.update(replace(q.performer, principal_id=q.reviewer.id))
    issue = q.command(issue, "assign", {"assignee_id": str(q.performer.id), "note": "Assigned to linked reviewer"})
    corrector = Principal.local(subject="unassigned-corrector", display_name="Other corrector")
    q.service.identities.save(corrector)
    q.service.assign_project_role(principal=q.owner, project=q.project, target_principal_id=corrector.id,
        role=ProjectRole.CORRECTOR, expected_revision=0, idempotency_key="other-corrector-acl")
    with pytest.raises(AuthorizationError):
        q.command(issue, "correct", {"evidence_version_ids": [str(q.corrected.id)], "note": "Not my assignment"},
                    principal=corrector)
    corrected = q.command(issue, "correct", {"evidence_version_ids": [str(q.corrected.id)], "note": "Own correction"},
                            principal=q.reviewer)
    assert corrected.state is QualityState.CORRECTED


def test_foreign_project_evidence_and_review_references_are_rejected(quality, tmp_path):
    from kraken_manager.domain.artifacts import ArtifactScope
    from kraken_manager.domain.project import GridOrientation
    q = quality
    other = q.service.create_project(principal=q.owner, name="Another project", width=1, height=1,
                                      orientation=GridOrientation.Y_DOWN, idempotency_key="other-project")
    series = q.service.create_artifact_series(principal=q.owner, project_id=other.id,
        scope=ArtifactScope.PROJECT_ATTACHMENT, name="Foreign report", idempotency_key="foreign-series")
    source = tmp_path / "foreign.txt"
    source.write_text("Another project's report")
    foreign = q.service.add_managed_artifact_version(principal=q.owner, project_id=other.id,
        series_id=series.id, source=source, idempotency_key="foreign-version")
    with pytest.raises(DomainValidationError, match="another project"):
        q.create(source_version_ids=[str(foreign.id)])
    with pytest.raises(DomainValidationError, match="review batch"):
        q.create(review_batch_id=str(uuid4()))
    assert q.service.list_quality_issues(q.project.id) == ()


def test_quality_write_failure_leaves_no_event_and_retry_can_succeed(quality, monkeypatch):
    q = quality
    original = FilesystemEventStore.append_preserved
    def fail(*_args, **_kwargs):
        raise OSError("injected event persistence failure")
    monkeypatch.setattr(FilesystemEventStore, "append_preserved", fail)
    with pytest.raises(OSError, match="injected"):
        q.create()
    monkeypatch.setattr(FilesystemEventStore, "append_preserved", original)
    assert q.service.list_quality_issues(q.project.id) == ()
    issue = q.create()
    assert len(q.service.quality_issue_history(q.project.id, issue.id)) == 1


def server_for(q):
    server = object.__new__(PostgresServerServices)
    server.projections = q.service._projection(q.project.id)
    server.events = FilesystemEventStore(q.service.catalog_root, q.project.id)
    server.uow_factory = q.service._uow(str(q.project.id))
    server.profiles = q.service.profiles
    server.performer_store = q.service.performers
    server.identities = q.service.identities
    server.project_ids_for_principal = lambda actor_id: (
        (str(q.project.id),) if q.service.identities.roles_for(q.project.id, actor_id) else ()
    )
    return server


def test_server_quality_facade_shares_local_events_and_translates_errors(quality):
    from kraken_server.services import ConflictError as ServerConflict, ForbiddenError, ValidationError
    q = quality
    server = server_for(q)
    ctx = CommandContext(str(q.owner.id), "server-quality", None)
    issue = server.transport_create_quality_issue(q.project.id, q.arguments, ctx)
    assert server.transport_create_quality_issue(q.project.id, q.arguments, ctx) == issue
    assert q.service.get_quality_issue(q.project.id, issue["id"]).revision == 0
    assigned = server.transport_mutate_quality_issue(q.project.id, issue["id"], {
        "operation": "assign", "arguments": {"assignee_id": str(q.performer.id), "note": "Fix"},
    }, CommandContext(str(q.owner.id), "server-assign", 0))
    assert assigned["state"] == "assigned"
    assert server.list_quality_issues(q.project.id, artifact_version_id=str(q.source.id)) == [assigned]
    with pytest.raises(ServerConflict):
        server.transport_mutate_quality_issue(q.project.id, issue["id"], {
            "operation": "exception", "arguments": {"note": "Stale"},
        }, CommandContext(str(q.owner.id), "stale", 0))
    with pytest.raises(ValidationError):
        server.transport_create_quality_issue(q.project.id, q.arguments | {"state": "accepted"},
                                             CommandContext(str(q.owner.id), "forged", None))
    outsider = Principal.local(subject="outsider", display_name="Outsider")
    q.service.identities.save(outsider)
    with pytest.raises(ForbiddenError):
        server.transport_mutate_quality_issue(q.project.id, issue["id"], {
            "operation": "exception", "arguments": {"note": "Forbidden"},
        }, CommandContext(str(outsider.id), "forbidden", 1))


def test_quality_routes_enforce_auth_revision_and_reverse_lookup(quality):
    from fastapi.testclient import TestClient
    from kraken_server.app import SessionPrincipal, create_app
    q = quality
    server = server_for(q)
    app = create_app(services=server, session_resolver=lambda token: SessionPrincipal(str(q.owner.id), "local", token))
    with TestClient(app) as client:
        base = f"/api/v1/projects/{q.project.id}/quality-issues"
        assert client.get(base).status_code == 401
        headers = {"Authorization": "Bearer owner", "Idempotency-Key": "http-quality"}
        created = client.post(base, json=q.arguments, headers=headers)
        assert created.status_code == 201, created.text
        issue = created.json()
        assert client.get(base, params={"artifact_version_id": str(q.source.id)}, headers=headers).json()["items"] == [issue]
        assert client.post(base + f"/{issue['id']}/commands", json={"operation": "exception",
            "arguments": {"note": "Design intent exception"}}, headers=headers).status_code == 422
        headers |= {"If-Match": "0", "Idempotency-Key": "http-exception"}
        response = client.post(base + f"/{issue['id']}/commands", json={"operation": "exception",
            "arguments": {"note": "Design intent exception"}}, headers=headers)
        assert response.status_code == 200, response.text
        assert response.json()["state"] == "exception"
        assert len(client.get(base + f"/{issue['id']}/history", headers=headers).json()["items"]) == 2


def test_selected_corrector_role_can_submit_correction_but_not_waive_issue(quality):
    from fastapi.testclient import TestClient
    from kraken_server.app import SessionPrincipal, create_app
    q = quality
    issue = q.create()
    issue = q.command(issue, "assign", {"assignee_id": str(q.performer.id), "note": "Repair"})
    corrector = Principal.local(subject="corrector", display_name="Corrector")
    q.service.identities.save(corrector)
    q.service.assign_project_role(principal=q.owner, project=q.project, target_principal_id=corrector.id,
        role=ProjectRole.CORRECTOR, expected_revision=0, idempotency_key="corrector-acl")
    server = server_for(q)
    app = create_app(services=server,
        session_resolver=lambda token: SessionPrincipal(str(corrector.id), "local", token))
    with TestClient(app) as client:
        url = f"/api/v1/projects/{q.project.id}/quality-issues/{issue.id}/commands"
        headers = {"Authorization": "Bearer corrector", "X-Kraken-Role": "corrector",
                   "Idempotency-Key": "corrector-result", "If-Match": "1"}
        response = client.post(url, json={"operation": "correct", "arguments": {
            "evidence_version_ids": [str(q.corrected.id)], "note": "Gap repaired"}}, headers=headers)
        assert response.status_code == 200, response.text
        assert response.json()["state"] == "corrected"
        headers |= {"Idempotency-Key": "forbidden-waiver", "If-Match": "2"}
        response = client.post(url, json={"operation": "exception", "arguments": {"note": "Cannot waive"}}, headers=headers)
        assert response.status_code == 403, response.text
