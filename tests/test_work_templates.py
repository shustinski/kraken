from dataclasses import replace
from uuid import uuid4
import json

import pytest

from kraken_manager.application.errors import AuthorizationError, ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.planning import WorkAssignment, WorkPlan, WorkScope
from kraken_manager.domain.planning_templates import (
    TemplateCriterion,
    TemplateDependency,
    TemplateTask,
    WorkPlanTemplate,
)
from tests.test_work_planning_integration import setup_project


def uid():
    return str(uuid4())


def recipe(project_id=None):
    return WorkPlanTemplate(
        uid(),
        project_id or uid(),
        1,
        "Layer reconstruction",
        (
            TemplateTask("prepare", "Prepare layer", "layer", effort_hours=1),
            TemplateTask("segment", "Segment image", "frame", effort_hours=0.5),
            TemplateTask("review", "Review mask", "frame", criteria=(TemplateCriterion("Engineer verified artifact"),)),
            TemplateTask("stitch", "Stitch layer", "layer"),
            TemplateTask("register", "Register layers", "project"),
        ),
        (
            TemplateDependency("prepare", "segment"),
            TemplateDependency("segment", "review"),
            TemplateDependency("review", "stitch"),
            TemplateDependency("stitch", "register"),
        ),
    )


def definition(template):
    return {k: v for k, v in template.to_dict()["template"].items() if k not in {"id", "project_id", "version"}}


def test_template_expansion_has_spatial_dependency_joins_and_atomic_provenance():
    template = recipe()
    scope = WorkScope(layer_ids=(uid(), uid()), frame_ids=(uid(), uid()))
    application_id = uid()
    batch = template.materialize(scope, application_id, assignments={"review": WorkAssignment(uid())})
    assert len(batch.items) == 13
    assert len(batch.dependencies) == 14
    assert batch == template.materialize(scope, application_id, assignments={"review": batch.items[6].assignment})
    plan = WorkPlan.create(template.project_id, "Recovery")
    result = plan.apply_template(batch, expected_revision=0)
    assert result.revision == 1
    assert len(plan.items) == 0
    provenance = result.template_applications[0]
    assert provenance.template_id == template.id and provenance.template_version == 1
    assert provenance.template_digest == template.digest
    assert set(provenance.item_ids) == {i.id for i in result.items}
    assert WorkPlan.from_dict(json.loads(json.dumps(result.to_dict()))) == result
    for edge in batch.dependencies:
        first, second = result.item(edge.predecessor_id), result.item(edge.successor_id)
        if first.scope.layer_ids and len(second.scope.layer_ids) == 1:
            assert first.scope.layer_ids == second.scope.layer_ids
        if len(first.scope.frame_ids) == len(second.scope.frame_ids) == 1:
            assert first.scope.frame_ids == second.scope.frame_ids
    with pytest.raises(DomainValidationError):
        result.apply_template(batch, expected_revision=1)
    assert result.revision == 1


def test_hierarchy_can_group_layer_work_under_project_stage():
    template = WorkPlanTemplate(
        uid(),
        uid(),
        1,
        "Grouped",
        (
            TemplateTask("phase", "Phase", "project"),
            TemplateTask("child", "Layer", "layer", parent_key="phase"),
        ),
    )
    batch = template.materialize(WorkScope(layer_ids=(uid(), uid())), uid())
    parent = next(i for i in batch.items if i.parent_id is None)
    assert all(i.parent_id == parent.id for i in batch.items if i.id != parent.id)
    plan = WorkPlan.create(template.project_id, "P").apply_template(batch, expected_revision=0)
    assert len(plan.items) == 3


@pytest.mark.parametrize(
    "tasks,edges",
    [
        ((TemplateTask("a", "A"), TemplateTask("a", "B")), ()),
        (
            (TemplateTask("a", "A"), TemplateTask("b", "B")),
            (TemplateDependency("a", "b"), TemplateDependency("b", "a")),
        ),
        ((TemplateTask("a", "A", parent_key="missing"),), ()),
        ((TemplateTask("a", "A", "frame"), TemplateTask("b", "B", "layer", parent_key="a")), ()),
        ((TemplateTask("a", "A"), TemplateTask("b", "B", parent_key="a")), (TemplateDependency("a", "b"),)),
    ],
)
def test_invalid_definitions_reject_cycles_duplicates_unknown_and_narrow_parent(tasks, edges):
    with pytest.raises(DomainValidationError):
        WorkPlanTemplate(uid(), uid(), 1, "Invalid", tasks, edges)


def test_expansion_requires_explicit_scope_and_is_bounded():
    template = recipe()
    with pytest.raises(DomainValidationError, match="layers"):
        template.materialize(WorkScope(), uid())
    with pytest.raises(DomainValidationError, match="frame selection"):
        template.materialize(WorkScope(layer_ids=(uid(),)), uid())
    with pytest.raises(DomainValidationError, match="batch limits"):
        template.materialize(WorkScope(layer_ids=(uid(),), frame_ids=tuple(uid() for _ in range(5000))), uid())
    with pytest.raises(DomainValidationError, match="assignments"):
        template.materialize(
            WorkScope(layer_ids=(uid(),), frame_ids=(uid(),)), uid(), assignments={"missing": WorkAssignment(uid())}
        )


def test_local_template_version_publish_materialize_replay_and_reopen(tmp_path):
    local, principal, project, layer, performer = setup_project(tmp_path)
    original = recipe(project.id)
    saved = local.save_work_template(
        project_id=project.id, principal=principal, definition=definition(original), idempotency_key="template"
    )
    assert saved == local.save_work_template(
        project_id=project.id, principal=principal, definition=definition(original), idempotency_key="template"
    )
    plan = local.create_work_plan(project_id=project.id, principal=principal, title="P", idempotency_key="plan")
    command = dict(
        project_id=project.id,
        plan_id=plan.id,
        principal=principal,
        expected_revision=0,
        operation="apply_template",
        idempotency_key="apply",
        arguments={
            "template_id": saved.id,
            "version": 1,
            "scope": {"layer_ids": [layer.id], "frame_coordinates": [[1, 1], [2, 1]]},
            "assignments": {t.key: {"performer_id": performer.id} for t in saved.tasks},
        },
    )
    changed = local.mutate_work_plan(**command)
    assert changed.revision == 1 and len(changed.items) == 7
    assert local.mutate_work_plan(**command) == changed
    updated_definition = definition(saved) | {
        "title": "Updated recipe",
        "tasks": [{"key": "new", "title": "New work", "multiplicity": "layer"}],
        "dependencies": [],
    }
    updated = local.save_work_template(
        project_id=project.id,
        principal=principal,
        template_id=saved.id,
        expected_version=1,
        definition=updated_definition,
        idempotency_key="template-v2",
    )
    assert updated.version == 2 and updated.digest != saved.digest
    assert local.get_work_template(project.id, saved.id, version=1) == saved
    assert local.get_work_template(project.id, saved.id) == updated
    assert local.list_work_templates(project.id) == (updated,)
    assert local.get_work_plan(project.id, plan.id) == changed
    assert local.mutate_work_plan(**command) == changed
    with pytest.raises(ConcurrencyError):
        local.save_work_template(
            project_id=project.id,
            principal=principal,
            template_id=saved.id,
            expected_version=1,
            definition=updated_definition,
            idempotency_key="stale",
        )
    with pytest.raises(ConflictError):
        local.save_work_template(
            project_id=project.id,
            principal=principal,
            definition=definition(original) | {"title": "Other"},
            idempotency_key="template",
        )
    from kraken_manager.infrastructure.filesystem import FilesystemEventStore

    events = FilesystemEventStore(local.catalog_root, project.id)
    assert len(events.load_stream(f"work-plan:{plan.id}")) == 2
    event = events.load_stream(f"work-plan:{plan.id}")[-1]
    assert event.event_type == "work_plan.apply_template"
    assert event.actor.principal_id == principal.id
    assert event.payload["command"]["arguments"]["materialization"]["template_digest"] == saved.digest
    from kraken_hub.composition import EmbeddedProjectService
    from tests.test_hub_embedded_service_parts.memory_secret_store import MemorySecretStoreMixin

    reopened = EmbeddedProjectService(local.data_dir, secret_store=MemorySecretStoreMixin._memory_secret_store())
    assert reopened.get_work_plan(project.id, plan.id) == changed
    assert reopened.get_work_template(project.id, saved.id, version=1) == saved


def test_materialization_rejects_invalid_references_and_unauthorized_actor_without_partial_writes(tmp_path):
    local, principal, project, layer, performer = setup_project(tmp_path)
    saved = local.save_work_template(
        project_id=project.id, principal=principal, definition=definition(recipe(project.id)), idempotency_key="t"
    )
    plan = local.create_work_plan(project_id=project.id, principal=principal, title="P", idempotency_key="p")
    args = {"template_id": saved.id, "version": 1, "scope": {"layer_ids": [layer.id], "frame_coordinates": [[1, 1]]}}
    command = dict(
        project_id=project.id,
        plan_id=plan.id,
        principal=principal,
        expected_revision=0,
        operation="apply_template",
        arguments=args,
        idempotency_key="apply",
    )
    for patch in (
        {"scope": {"layer_ids": [uid()], "frame_coordinates": [[1, 1]]}},
        {"assignments": {"review": {"performer_id": uid()}}},
        {"version": 99},
    ):
        with pytest.raises((DomainValidationError, NotFoundError)):
            local.mutate_work_plan(**(command | {"arguments": args | patch}))
        assert local.get_work_plan(project.id, plan.id) == plan
    stranger = Principal.local(subject="stranger", display_name="Stranger")
    local.identities.save(stranger)
    with pytest.raises(AuthorizationError):
        local.mutate_work_plan(**(command | {"principal": stranger}))
    with pytest.raises(AuthorizationError):
        local.save_work_template(
            project_id=project.id,
            principal=stranger,
            definition=definition(recipe(project.id)),
            idempotency_key="unauthorized",
        )
    assert local.get_work_plan(project.id, plan.id) == plan


def test_template_definition_cannot_inject_runtime_state(tmp_path):
    local, principal, project, layer, performer = setup_project(tmp_path)
    for bad in ("state", "progress", "assignment", "plugin_job_ids", "scope", "id"):
        value = {"title": "Bad", "tasks": [{"key": "x", "title": "X", bad: "accepted"}]}
        with pytest.raises(DomainValidationError):
            local.save_work_template(project_id=project.id, principal=principal, definition=value, idempotency_key=bad)
    assert local.list_work_templates(project.id) == ()


def http_remote(tmp_path):
    from fastapi.testclient import TestClient
    from kraken_server.app import SessionPrincipal, create_app
    from kraken_hub.infrastructure.remote.parts.work_planning import WorkPlanningMixin as RemotePlanning
    from kraken_manager.infrastructure.filesystem import FilesystemEventStore
    from tests.test_work_planning_integration import PlanningServer

    local, principal, project, layer, performer = setup_project(tmp_path)
    server = object.__new__(PlanningServer)
    server.projections = local._projection(project.id)
    server.events = FilesystemEventStore(local.catalog_root, project.id)
    server.uow_factory = local._uow(str(project.id))
    server.profiles, server.performer_store, server.identities = local.profiles, local.performers, local.identities
    server._actor = lambda actor_id: principal
    server.project_ids_for_principal = lambda actor_id: (
        (str(project.id),) if local.identities.roles_for(project.id, actor_id) else ()
    )
    client = TestClient(
        create_app(
            services=server,
            session_resolver=lambda token: SessionPrincipal(principal.id, "local", token),
            development=True,
        )
    )

    class Remote(RemotePlanning):
        def _call(self, method, path, *, payload=None, idempotency_key=None, if_match=None):
            headers = {"Authorization": "Bearer test"}
            if idempotency_key:
                headers["Idempotency-Key"] = idempotency_key
            if if_match is not None:
                headers["If-Match"] = str(if_match)
            response = client.request(method, path, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    return local, principal, project, layer, performer, server, client, Remote()


def test_remote_http_template_versioning_and_atomic_work_generation(tmp_path):
    local, principal, project, layer, performer, server, client, remote = http_remote(tmp_path)
    template = remote.save_work_template(
        project_id=project.id,
        principal=principal,
        definition=definition(recipe(project.id)),
        idempotency_key="http-template",
    )
    assert template.version == 1
    assert remote.list_work_templates(project.id) == (template,)
    assert remote.get_work_template(project.id, template.id, version=1) == template
    plan = remote.create_work_plan(
        project_id=project.id, principal=principal, title="HTTP", idempotency_key="http-plan"
    )
    changed = remote.mutate_work_plan(
        project_id=project.id,
        plan_id=plan.id,
        principal=principal,
        expected_revision=0,
        operation="apply_template",
        arguments={
            "template_id": template.id,
            "version": 1,
            "scope": {"layer_ids": [layer.id], "frame_coordinates": [[1, 1], [2, 1]]},
        },
        idempotency_key="http-apply",
    )
    assert len(changed.items) == 7 and changed.revision == 1
    assert remote.get_work_plan(project.id, plan.id) == local.get_work_plan(project.id, plan.id) == changed
    updated = remote.save_work_template(
        project_id=project.id,
        principal=principal,
        template_id=template.id,
        expected_version=1,
        definition=definition(template) | {"title": "Second version"},
        idempotency_key="http-version2",
    )
    assert updated.version == 2
    assert remote.get_work_template(project.id, template.id, version=1) == template
    assert remote.list_work_templates(project.id) == (updated,)
    headers = {"Authorization": "Bearer test", "Idempotency-Key": "stale", "If-Match": "0"}
    stale = client.put(
        f"/api/v1/projects/{project.id}/work-plan-templates/{template.id}", headers=headers, json=definition(template)
    )
    assert stale.status_code == 409
    forged = client.post(
        f"/api/v1/projects/{project.id}/work-plans/{plan.id}/commands",
        headers=headers | {"Idempotency-Key": "forged", "If-Match": "1"},
        json={"operation": "apply_template", "arguments": {"materialization": {"items": []}}},
    )
    assert forged.status_code == 422
    assert remote.get_work_plan(project.id, plan.id) == changed


def test_templates_materialize_once_under_concurrent_same_key_retries(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    local, principal, project, layer, performer = setup_project(tmp_path)
    template = local.save_work_template(
        project_id=project.id, principal=principal, definition=definition(recipe(project.id)), idempotency_key="t"
    )
    plan = local.create_work_plan(project_id=project.id, principal=principal, title="P", idempotency_key="p")
    barrier = Barrier(2)

    def execute(_):
        barrier.wait(timeout=5)
        return local.mutate_work_plan(
            project_id=project.id,
            plan_id=plan.id,
            principal=principal,
            expected_revision=0,
            operation="apply_template",
            arguments={
                "template_id": template.id,
                "version": 1,
                "scope": {"layer_ids": [layer.id], "frame_coordinates": [[1, 1]]},
            },
            idempotency_key="same",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, (1, 2)))
    assert results[0] == results[1]
    assert local.get_work_plan(project.id, plan.id).revision == 1
    from kraken_manager.infrastructure.filesystem import FilesystemEventStore

    events = FilesystemEventStore(local.catalog_root, project.id)
    assert len(events.load_stream(f"work-plan:{plan.id}")) == 2


def test_global_template_stream_cannot_be_updated_from_another_project(tmp_path):
    from kraken_manager.application.planning_templates import WorkTemplateService
    from kraken_manager.domain.identity import ProjectRole

    local, principal, project, layer, performer = setup_project(tmp_path)
    template = recipe(project.id)

    class GlobalEvents:
        events = ()

        def load_stream(self, stream_id):
            return self.events

        def current_revision(self, stream_id):
            return len(self.events)

        def find_by_idempotency_key(self, project_id, key):
            return tuple(e for e in self.events if e.project_id == project_id and e.idempotency_key == key)

        def append(self, stream_id, *, expected_revision, events):
            self.events += tuple(events)

    events = GlobalEvents()
    service = WorkTemplateService(events)
    context = dict(principal=principal, storage=local.profile, roles=(ProjectRole.ADMIN,))
    service.publish(template, expected_version=0, idempotency_key="t", **context)
    with pytest.raises(NotFoundError, match="project"):
        service.publish(
            replace(template, project_id=uid(), version=2), expected_version=1, idempotency_key="hijack", **context
        )
    assert len(events.events) == 1


def test_template_replay_rechecks_permissions_and_legacy_plan_payload_still_loads(tmp_path):
    local, principal, project, layer, performer = setup_project(tmp_path)
    source = definition(recipe(project.id))
    template = local.save_work_template(
        project_id=project.id, principal=principal, definition=source, idempotency_key="t"
    )
    plan = local.create_work_plan(project_id=project.id, principal=principal, title="P", idempotency_key="p")
    payload = plan.to_dict()
    payload["plan"].pop("template_applications")
    assert WorkPlan.from_dict(payload) == plan
    stranger = Principal.local(subject="revoked", display_name="Revoked")
    local.identities.save(stranger)
    with pytest.raises(AuthorizationError):
        local.save_work_template(project_id=project.id, principal=stranger, definition=source, idempotency_key="t")
    command = dict(
        project_id=project.id,
        plan_id=plan.id,
        principal=principal,
        expected_revision=0,
        operation="apply_template",
        arguments={
            "template_id": template.id,
            "version": 1,
            "scope": {"layer_ids": [layer.id], "frame_coordinates": [[1, 1]]},
        },
        idempotency_key="apply",
    )
    changed = local.mutate_work_plan(**command)
    with pytest.raises(AuthorizationError):
        local.mutate_work_plan(**(command | {"principal": stranger}))
    assert local.get_work_plan(project.id, plan.id) == changed
