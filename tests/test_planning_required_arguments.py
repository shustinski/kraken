"""Incomplete client commands are validation errors and leave no state change."""
import pytest

from kraken_manager.domain.common import DomainValidationError
from tests.test_work_planning_integration import setup_project

pytest_plugins = ["tests.test_admin_project_workflows"]


@pytest.mark.parametrize("operation,arguments", [
    ("add_item", {}), ("apply_template", {}), ("assign", {}), ("revise_item", {}),
    ("add_dependency", {}), ("add_milestone", {}), ("transition", {}), ("move_item", {}),
    ("record_progress", {}), ("satisfy_criterion", {}), ("link_execution", {}),
    ("record_progress", {"item_id": "00000000-0000-0000-0000-000000000001", "fraction": 1}),
    ("add_item", {"item": {}}), ("add_item", {"item": {"title": "Incomplete", "assignment": {}}}),
    ("add_item", {"item": {"title": "Incomplete", "criteria": [{}]}}),
    ("add_dependency", {"dependency": {}}), ("add_milestone", {"milestone": {}}),
])
def test_missing_required_command_fields_are_domain_validation(operation, arguments, tmp_path):
    local, actor, project, *_ = setup_project(tmp_path)
    plan = local.create_work_plan(project_id=project.id, title="Validation", principal=actor, idempotency_key="plan")
    before = local.history(project.id)
    with pytest.raises(DomainValidationError, match="Missing"):
        local.mutate_work_plan(project_id=project.id, plan_id=plan.id, principal=actor,
            expected_revision=plan.revision, operation=operation, arguments=arguments, idempotency_key="incomplete")
    assert local.get_work_plan(project.id, plan.id) == plan
    assert local.history(project.id) == before


def test_postgres_http_invalid_commands_return_422_and_same_key_can_be_corrected(database):
    from fastapi.testclient import TestClient
    from kraken_server.app import SessionPrincipal, create_app
    from kraken_server.services import CommandContext
    from tests.test_admin_project_workflows import project

    service, _, actor, _ = database
    pid = project(service, actor)["project_id"]
    plan = service.transport_create_work_plan(pid, {"title": "HTTP validation"}, CommandContext(actor.id, "plan", None))
    client = TestClient(create_app(services=service,
        session_resolver=lambda token: SessionPrincipal(str(actor.id), "local", token) if token == "owner" else None))
    headers = {"Authorization": "Bearer owner", "If-Match": "0", "Idempotency-Key": "invalid-then-corrected"}
    url = f"/api/v1/projects/{pid}/work-plans/{plan['id']}/commands"
    for operation in ("add_item", "record_progress", "satisfy_criterion"):
        response = client.post(url, headers=headers, json={"operation": operation, "arguments": {}})
        assert response.status_code == 422 and "Missing" in response.text, response.text
        assert service.get_work_plan(pid, plan["id"]) == plan
    response = client.post(url, headers=headers, json={"operation": "add_item", "arguments": {"item": {"title": "Corrected request"}}})
    assert response.status_code == 200 and response.json()["revision"] == 1, response.text
    assert len(service.events.load_stream(f"work-plan:{plan['id']}")) == 2
