from types import SimpleNamespace
from uuid import uuid4

import pytest

from kraken_manager.application.errors import ConflictError
from kraken_manager.application.quality import QualityService
from kraken_hub.infrastructure.remote.client import RemoteServerProjectService
from kraken_hub.composition.catalog import DualCatalogService
from tests.test_quality_workflow import quality as quality, server_for


def test_remote_quality_and_dual_catalog_use_server_identity_and_contract(quality, monkeypatch):
    from fastapi.testclient import TestClient
    from kraken_server.app import SessionPrincipal, create_app
    q = quality
    app = create_app(services=server_for(q), session_resolver=lambda token: SessionPrincipal(str(q.owner.id), "local", token))
    remote = object.__new__(RemoteServerProjectService)
    remote.auth = SimpleNamespace(principal=q.owner)
    with TestClient(app) as client:
        def call(method, path, *, payload=None, idempotency_key=None, if_match=None, **kwargs):
            headers = {"Authorization": "Bearer owner"}
            if idempotency_key is not None:
                headers["Idempotency-Key"] = idempotency_key
            if if_match is not None:
                headers["If-Match"] = str(if_match)
            response = client.request(method, path, json=payload, headers=headers)
            assert response.status_code < 400, response.text
            return response.json()
        remote._call = call
        remote.list_projects = lambda **kwargs: (q.project,)
        dual = DualCatalogService(q.service, remote)
        monkeypatch.setattr(q.service, "create_quality_issue", lambda **kwargs: pytest.fail("Remote write routed locally"))
        created = dual.create_quality_issue(project_id=q.project.id, principal=q.reviewer,
            arguments=q.arguments, idempotency_key="remote-quality")
        assert created.created_by == str(q.owner.id)
        assert dual.list_quality_issues(q.project.id, artifact_version_id=str(q.source.id)) == (created,)
        assert dual.get_quality_issue(q.project.id, created.id) == created
        waived = dual.mutate_quality_issue(project_id=q.project.id, principal=q.reviewer, issue_id=created.id,
            expected_revision=0, operation="exception", arguments={"note": "Authorized remote waiver"}, idempotency_key="waive")
        assert waived.revision == 1
        assert len(dual.quality_issue_history(q.project.id, created.id)) == 2


def test_quality_gate_uses_actual_versions_pending_events_and_authorized_exceptions(quality):
    q = quality
    issue = q.create()
    with q.service._uow(str(q.project.id))() as unit:
        assert QualityService.blocking_issues(unit.event_store, q.project.id, [str(q.source.id)]) == (issue,)
        assert QualityService.blocking_issues(unit.event_store, q.project.id, [str(uuid4())]) == ()
        with pytest.raises(ConflictError, match=issue.id):
            QualityService.require_clear_inputs(unit.event_store, q.project.id, [str(q.source.id)])
        QualityService.execute(unit, profiles=q.service.profiles, performers=q.service.performers,
            project_id=q.project.id, principal=q.owner, issue_id=issue.id, expected_revision=0,
            operation="exception", arguments={"note": "Reviewed waiver"}, idempotency_key="pending-waiver")
        assert QualityService.blocking_issues(unit.event_store, q.project.id, [str(q.source.id)]) == ()
        # Deliberately no commit: pending waiver cannot unblock a later transaction.
    with q.service._uow(str(q.project.id))() as unit:
        assert len(QualityService.blocking_issues(unit.event_store, q.project.id, [str(q.source.id)])) == 1


def test_plan_criterion_acceptance_is_gated_by_matching_quality_scope(quality):
    q = quality
    issue = q.create()
    plan = q.service.create_work_plan(project_id=q.project.id, title="Restore layer", principal=q.owner,
                                      idempotency_key="plan")
    plan = q.service.mutate_work_plan(project_id=q.project.id, plan_id=plan.id, principal=q.owner,
        expected_revision=0, operation="add_item", arguments={"item": {"title": "Accept conductor",
            "scope": {"layer_ids": [str(q.layer.id)]}, "assignment": {"performer_id": str(q.performer.id)},
            "criteria": [{"description": "Continuity"}]}},
        idempotency_key="plan-item")
    item = plan.items[0]
    plan = q.service.mutate_work_plan(project_id=q.project.id, plan_id=plan.id, principal=q.owner,
        expected_revision=plan.revision, operation="transition", arguments={"state": "active"}, idempotency_key="activate-plan")
    for state in ("running", "in_review"):
        if state == "in_review":
            plan = q.service.mutate_work_plan(project_id=q.project.id, plan_id=plan.id, principal=q.owner,
                expected_revision=plan.revision, operation="record_progress", arguments={"item_id": item.id,
                    "fraction": 1, "actual_effort_hours": 1}, idempotency_key="progress")
        plan = q.service.mutate_work_plan(project_id=q.project.id, plan_id=plan.id, principal=q.owner,
            expected_revision=plan.revision, operation="move_item", arguments={"item_id": item.id, "state": state},
            idempotency_key=f"item-{state}")
    command = dict(project_id=q.project.id, plan_id=plan.id, principal=q.reviewer, expected_revision=plan.revision,
        operation="satisfy_criterion", arguments={"item_id": item.id, "criterion_id": item.criteria[0].id,
            "evidence_version_ids": [str(q.corrected.id)]}, idempotency_key="criterion")
    with pytest.raises(ConflictError, match=issue.id):
        q.service.mutate_work_plan(**command)
    q.command(issue, "exception", {"note": "Approved design exception"}, principal=q.reviewer)
    assert q.service.mutate_work_plan(**command).items[0].criteria[0].satisfied
