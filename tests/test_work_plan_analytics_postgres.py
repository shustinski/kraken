from datetime import UTC, datetime

from kraken_server.services import CommandContext
from tests.test_admin_project_workflows import project

pytest_plugins = ["tests.test_admin_project_workflows"]


def test_postgres_reports_are_historical_versioned_and_do_not_mutate_plans(database):
    services, accounts, principal, config = database
    project_id = project(services, principal)["project_id"]
    plan = services.transport_create_work_plan(
        project_id, {"title": "Measured"}, CommandContext(principal.id, "plan", None)
    )
    services.transport_mutate_work_plan(
        project_id,
        plan["id"],
        {"operation": "add_item", "arguments": {"item": {"title": "A", "estimate": {"effort_hours": 5}}}},
        CommandContext(principal.id, "a", 0),
    )
    before = datetime.now(UTC)
    second = services.transport_mutate_work_plan(
        project_id,
        plan["id"],
        {"operation": "add_item", "arguments": {"item": {"title": "B"}}},
        CommandContext(principal.id, "b", 1),
    )
    report = services.transport_work_plan_analytics(project_id, {}, principal.id)
    assert report["source_plan_revisions"] == {plan["id"]: 2}
    assert report["summary"]["leaf_items"] == 2
    assert report["forecast"]["execution_finish_at"] is None
    historical = services.transport_work_plan_analytics(project_id, {"as_of": before.isoformat()}, principal.id)
    assert historical["source_plan_revisions"] == {plan["id"]: 1}
    assert historical["summary"]["leaf_items"] == 1
    assert services.get_work_plan(project_id, plan["id"]) == second
    assert len(services.events.load_stream(f"work-plan:{plan['id']}")) == 3
