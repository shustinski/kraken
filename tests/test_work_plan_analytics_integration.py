from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from kraken_manager.application.errors import AuthorizationError
from tests.test_work_planning_integration import setup_project
from tests.test_work_templates import http_remote


def seed_plan(local, principal, project, layer, performer):
    plan = local.create_work_plan(
        project_id=project.id, principal=principal, title="Measured recovery", idempotency_key=str(uuid4())
    )

    def change(operation, arguments):
        nonlocal plan
        plan = local.mutate_work_plan(
            project_id=project.id,
            plan_id=plan.id,
            principal=principal,
            expected_revision=plan.revision,
            operation=operation,
            arguments=arguments,
            idempotency_key=str(uuid4()),
        )
        return plan

    change(
        "add_item",
        {
            "item": {
                "title": "Mask review",
                "scope": {"layer_ids": [layer.id], "frame_coordinates": [[1, 1]]},
                "assignment": {"performer_id": performer.id},
                "estimate": {"effort_hours": 8},
            }
        },
    )
    first = plan.items[-1].id
    change(
        "add_item",
        {
            "item": {
                "title": "Microscope review",
                "scope": {"layer_ids": [layer.id]},
                "assignment": {"performer_id": performer.id},
                "estimate": {"effort_hours": 4},
            }
        },
    )
    second = plan.items[-1].id
    change("capture_baseline", {})
    change("revise_item", {"item_id": second, "estimate": {"effort_hours": 6}})
    change("transition", {"state": "active"})
    change("move_item", {"item_id": first, "state": "running"})
    change("record_progress", {"item_id": first, "fraction": 1, "actual_effort_hours": 3})
    change("move_item", {"item_id": first, "state": "in_review"})
    before_acceptance = datetime.now(UTC)
    historical_revision = plan.revision
    change("move_item", {"item_id": first, "state": "accepted"})
    change("move_item", {"item_id": second, "state": "blocked", "reason": "Awaiting microscope"})
    return plan, first, second, before_acceptance, historical_revision


def test_local_reports_capture_versions_real_events_names_and_historical_state(tmp_path):
    local, principal, project, layer, performer = setup_project(tmp_path)
    plan, first, second, before, prior_revision = seed_plan(local, principal, project, layer, performer)
    report = local.work_plan_analytics(project.id, principal=principal)
    assert report["source_plan_revisions"] == {plan.id: plan.revision}
    assert report["summary"]["readiness_fraction"] == 0.5
    assert report["summary"]["reported_effort_hours"] == 3
    assert report["performer_names"][performer.id] == performer.name
    assert report["layer_names"][layer.id] == layer.name
    assert report["summary"]["mean_lead_hours"] >= report["summary"]["mean_cycle_hours"] >= 0
    assert report["summary"]["measured_compute_utilization"] is None
    assert report["forecast"]["reason"] == "blocked_work"
    historical = local.work_plan_analytics(project.id, principal=principal, as_of=before)
    assert historical["source_plan_revisions"][plan.id] == prior_revision
    assert historical["summary"]["accepted_items"] == 0
    assert historical["state_counts"] == {"in_review": 1, "planned": 1}
    with pytest.raises(AuthorizationError):
        local.work_plan_analytics(project.id, principal=replace(principal, active=False))


def test_http_remote_analytics_matches_authorized_local_projection_and_filters(tmp_path, monkeypatch):
    sample = {"scope": "whole_host", "cpu_percent": 12, "memory": {"total_bytes": 1024}}
    monkeypatch.setattr("kraken_agent.telemetry.sample_host", lambda: sample)
    local, principal, project, layer, performer, server, client, remote = http_remote(tmp_path)
    plan, first, second, before, prior_revision = seed_plan(local, principal, project, layer, performer)
    as_of = datetime.now(UTC)
    request = dict(as_of=as_of, start=as_of - timedelta(days=1), end=as_of, capacity_by_performer={performer.id: 6})
    local_report = local.work_plan_analytics(project.id, principal=principal, **request)
    remote_report = remote.work_plan_analytics(project.id, principal=principal, **request)
    local_compute = local_report.pop("current_compute_resources")
    remote_compute = remote_report.pop("current_compute_resources")
    assert local_compute == {
        "admission": {"state": "unenforced", "admission_ready": False, "scope": "local_agent",
                      "os_enforced": False, "per_host_capacity_enforced": False},
        "host_observations": [{"sample": sample, "stale": False, "age_seconds": 0}],
        "time_basis": "current_observation_not_report_as_of",
    }
    # This fixture intentionally has no server Agent gateway. It must not report
    # the Desktop host's observation as the server's resource availability.
    assert remote_compute == {
        "admission": {"state": "unavailable", "admission_ready": False},
        "host_observations": [], "time_basis": "current_observation_not_report_as_of",
    }
    assert remote_report == local_report
    filtered = remote.work_plan_analytics(project.id, principal=principal, states=["blocked"])
    assert filtered["state_counts"] == {"blocked": 1}
    assert filtered["items"][0]["item_id"] == second
    assert filtered["baseline_variance"]
    endpoint = f"/api/v1/projects/{project.id}/work-planning/analytics"
    headers = {"Authorization": "Bearer test"}
    malformed = client.post(endpoint, headers=headers, json={"capacity_by_performer": {performer.id: -1}})
    assert malformed.status_code == 422, malformed.text
    forbidden = client.post(f"/api/v1/projects/{uuid4()}/work-planning/analytics", headers=headers, json={})
    assert forbidden.status_code == 403, forbidden.text
    assert local.get_work_plan(project.id, plan.id) == plan
