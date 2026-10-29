"""Durable Hub/server planning scenarios against real isolated project stores."""

from uuid import uuid4

import pytest

from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.composition.parts.work_planning import WorkPlanningMixin as LocalPlanning
from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.project import GridOrientation, LayerType
from kraken_manager.infrastructure.filesystem import FilesystemEventStore
from kraken_server.persistent_services import PostgresServerServices
from kraken_server.persistent_services_parts.work_planning import WorkPlanningMixin as ServerPlanning
from kraken_server.services import CommandContext
from tests.test_hub_embedded_service_parts.memory_secret_store import MemorySecretStoreMixin


PlanningService = (
    EmbeddedProjectService
    if issubclass(EmbeddedProjectService, LocalPlanning)
    else type("PlanningService", (LocalPlanning, EmbeddedProjectService), {})
)


PlanningServer = (
    PostgresServerServices
    if issubclass(PostgresServerServices, ServerPlanning)
    else type("PlanningServer", (ServerPlanning, PostgresServerServices), {})
)


def setup_project(tmp_path):
    local = PlanningService(tmp_path / "data", secret_store=MemorySecretStoreMixin._memory_secret_store())
    session = local.create_initial_account("owner", "Owner", "")
    project = local.create_project(
        principal=session.principal,
        name="Recovery",
        width=2,
        height=2,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key="project",
    )
    layer = local.create_layer(
        principal=session.principal,
        project=project,
        name="M1",
        layer_type=LayerType.METAL,
        order=0,
        idempotency_key="layer",
    )
    performer = local.create_manual_performer(name="Engineer", color="#123456")
    return local, session.principal, project, layer, performer


def test_hub_plan_create_replay_add_item_reopen_and_scope_validation(tmp_path):
    local, principal, project, layer, performer = setup_project(tmp_path)
    create = dict(project_id=project.id, title="Restore M1", principal=principal, idempotency_key="plan")
    plan = local.create_work_plan(**create)
    assert local.create_work_plan(**create) == plan
    args = {
        "item": {
            "title": "Trace mask",
            "scope": {"layer_ids": [layer.id], "frame_coordinates": [[1, 2]]},
            "assignment": {"performer_id": performer.id},
            "criteria": [{"description": "Reviewed mask"}],
        }
    }
    command = dict(
        project_id=project.id,
        plan_id=plan.id,
        principal=principal,
        expected_revision=0,
        operation="add_item",
        arguments=args,
        idempotency_key="item",
    )
    plan = local.mutate_work_plan(**command)
    assert local.mutate_work_plan(**command) == plan
    assert plan.items[0].scope.frame_ids == (project.frame_id_at(1, 2),)
    assert local.list_work_plans(project.id) == (plan,)
    reopened = PlanningService(local.data_dir, secret_store=MemorySecretStoreMixin._memory_secret_store())
    assert reopened.get_work_plan(project.id, plan.id) == plan
    event_count = FilesystemEventStore(local.catalog_root, project.id).last_global_position()
    for forbidden in ("state", "progress", "work_order_ids", "attempt"):
        bad_args = {"item": {"title": "Forged", forbidden: "accepted"}}
        with pytest.raises(DomainValidationError):
            local.mutate_work_plan(
                **(
                    command
                    | {
                        "expected_revision": plan.revision,
                        "arguments": bad_args,
                        "idempotency_key": f"forged-{forbidden}",
                    }
                )
            )
    assert FilesystemEventStore(local.catalog_root, project.id).last_global_position() == event_count
    foreign = str(uuid4())
    with pytest.raises(DomainValidationError, match="Scope layer"):
        local.mutate_work_plan(
            **(
                command
                | {
                    "expected_revision": plan.revision,
                    "idempotency_key": "foreign",
                    "arguments": {"item": {"title": "Foreign", "scope": {"layer_ids": [foreign]}}},
                }
            )
        )


def test_server_facade_and_hub_share_persisted_plans_and_enforce_revisions(tmp_path):
    from kraken_server.services import ConflictError, ValidationError

    local, principal, project, layer, performer = setup_project(tmp_path)
    server = object.__new__(PlanningServer)
    server.projections = local._projection(project.id)
    server.events = FilesystemEventStore(local.catalog_root, project.id)
    server.uow_factory = local._uow(str(project.id))
    server.profiles = local.profiles
    server.performer_store = local.performers
    server._actor = lambda actor_id: principal
    context = CommandContext(principal.id, "server-plan", 0)
    record = server.transport_create_work_plan(project.id, {"title": "Server plan"}, context)
    assert record == server.transport_create_work_plan(project.id, {"title": "Server plan"}, context)
    mutation = {
        "operation": "add_item",
        "arguments": {
            "item": {
                "title": "Segment",
                "assignment": {"performer_id": performer.id},
                "scope": {"layer_ids": [layer.id]},
            }
        },
    }
    context = CommandContext(principal.id, "server-item", 0)
    changed = server.transport_mutate_work_plan(project.id, record["id"], mutation, context)
    assert changed["revision"] == 1
    assert server.transport_mutate_work_plan(project.id, record["id"], mutation, context) == changed
    assert local.get_work_plan(project.id, record["id"]).revision == 1
    assert server.list_work_plans(project.id) == [changed]
    assert server.get_work_plan(project.id, record["id"]) == changed
    with pytest.raises(ConflictError):
        server.transport_mutate_work_plan(project.id, record["id"], mutation, CommandContext(principal.id, "stale", 0))
    with pytest.raises(ValidationError):
        server.transport_mutate_work_plan(
            project.id,
            record["id"],
            {"operation": "capture_baseline", "arguments": {"at": "2000"}},
            CommandContext(principal.id, "bad-time", 1),
        )


def test_decode_acceptance_evidence_checks_project_layer_and_frame():
    from types import SimpleNamespace
    from kraken_manager.domain.planning import AcceptanceCriterion, WorkItem, WorkPlan, WorkScope
    from kraken_manager.domain.project import Project
    from kraken_manager.infrastructure.planning_commands import decode_arguments

    project = Project.create(name="P", width=2, height=2, storage_profile="test")
    layer, other_layer, evidence_id, series_id = (str(uuid4()) for _ in range(4))
    item = WorkItem.create(
        "Review",
        scope=WorkScope(layer_ids=(layer,), frame_ids=(project.frame_id_at(1, 1),)),
        criteria=(AcceptanceCriterion(str(uuid4()), "Mask accepted"),),
    )
    plan = WorkPlan.create(project.id, "P").add_item(item, expected_revision=0)
    series = SimpleNamespace(project_id=project.id, layer_id=layer, frame_id=project.frame_id_at(1, 1), archived=False)
    projections = SimpleNamespace(
        get_artifact_version=lambda key: SimpleNamespace(series_id=series_id), get_artifact_series=lambda key: series
    )
    args = {"item_id": item.id, "criterion_id": item.criteria[0].id, "evidence_version_ids": [evidence_id]}
    assert decode_arguments("satisfy_criterion", args, projections, project, None, "accept", plan=plan)[
        "evidence_version_ids"
    ] == (evidence_id,)
    for name, value in (
        ("project_id", str(uuid4())),
        ("layer_id", other_layer),
        ("frame_id", project.frame_id_at(2, 2)),
    ):
        previous = getattr(series, name)
        setattr(series, name, value)
        with pytest.raises(DomainValidationError, match="outside"):
            decode_arguments("satisfy_criterion", args, projections, project, None, "accept", plan=plan)
        setattr(series, name, previous)
    with pytest.raises(DomainValidationError, match="fields"):
        decode_arguments(
            "satisfy_criterion", args | {"reviewer_id": str(uuid4())}, projections, project, None, "accept", plan=plan
        )


def test_server_invalid_transition_reports_conflict_without_writing_event(tmp_path):
    from kraken_server.services import ConflictError

    local, principal, project, layer, performer = setup_project(tmp_path)
    server = object.__new__(PlanningServer)
    server.projections = local._projection(project.id)
    server.events = FilesystemEventStore(local.catalog_root, project.id)
    server.uow_factory = local._uow(str(project.id))
    server.profiles = local.profiles
    server.performer_store = local.performers
    server._actor = lambda actor_id: principal
    record = server.transport_create_work_plan(project.id, {"title": "P"}, CommandContext(principal.id, "plan", 0))
    before = server.events.last_global_position()
    with pytest.raises(ConflictError):
        server.transport_mutate_work_plan(
            project.id,
            record["id"],
            {"operation": "transition", "arguments": {"state": "completed"}},
            CommandContext(principal.id, "bad-transition", 0),
        )
    assert server.events.last_global_position() == before


def test_http_planning_endpoints_persist_and_report_revision_conflicts(tmp_path):
    from fastapi.testclient import TestClient
    from kraken_server.app import SessionPrincipal, create_app

    local, principal, project, layer, performer = setup_project(tmp_path)
    server = object.__new__(PlanningServer)
    server.projections = local._projection(project.id)
    server.events = FilesystemEventStore(local.catalog_root, project.id)
    server.uow_factory = local._uow(str(project.id))
    server.profiles = local.profiles
    server.performer_store = local.performers
    server.identities = local.identities
    server.project_ids_for_principal = lambda actor_id: (
        (str(project.id),) if local.identities.roles_for(project.id, actor_id) else ()
    )
    server._actor = lambda actor_id: principal
    app = create_app(
        services=server, session_resolver=lambda token: SessionPrincipal(principal.id, "local", token), development=True
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer test", "Idempotency-Key": "http-plan"}
    base = f"/api/v1/projects/{project.id}/work-plans"
    response = client.post(base, headers=headers, json={"title": "HTTP recovery"})
    assert response.status_code == 201, response.text
    plan_id = response.json()["id"]
    assert client.post(base, headers=headers, json={"title": "HTTP recovery"}).json() == response.json()
    body = {"operation": "add_item", "arguments": {"item": {"title": "Segment"}}}
    response = client.post(
        f"{base}/{plan_id}/commands", headers=headers | {"Idempotency-Key": "http-item", "If-Match": "0"}, json=body
    )
    assert response.status_code == 200, response.text
    assert response.json()["revision"] == 1
    stale = client.post(
        f"{base}/{plan_id}/commands", headers=headers | {"Idempotency-Key": "http-stale", "If-Match": "0"}, json=body
    )
    assert stale.status_code == 409, stale.text
    assert client.get(base, headers=headers).json()["items"][0]["id"] == plan_id
    assert client.get(f"{base}/{plan_id}", headers=headers).json() == response.json()
