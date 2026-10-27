"""Templates on real isolated PostgreSQL schemas; no production storage is used."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from kraken_manager.infrastructure.postgres.event_store import PostgresEventStore
from kraken_server.services import CommandContext, ConflictError, NotFoundError
from tests.test_admin_project_workflows import project
from tests.test_work_templates import definition, recipe

pytest_plugins = ["tests.test_admin_project_workflows"]


def setup_template(database):
    services, accounts, principal, config = database
    record = project(services, principal)
    project_id = record["project_id"]
    layer = services.create_layer(
        project_id,
        {"name": "M1", "type": "metal", "order": 0},
        CommandContext(principal.id, "layer", record["revision"]),
    )
    template = services.transport_save_work_template(
        project_id, definition(recipe(project_id)), CommandContext(principal.id, "template", 0)
    )
    plan = services.transport_create_work_plan(project_id, {"title": "P"}, CommandContext(principal.id, "plan", None))
    args = {
        "template_id": template["id"],
        "version": 1,
        "scope": {"layer_ids": [layer["layer_id"]], "frame_coordinates": [[1, 1], [2, 1]]},
    }
    return services, principal, project_id, template, plan, args


def test_postgres_template_materialization_is_atomic_under_concurrent_plan_updates(database):
    services, principal, project_id, template, plan, args = setup_template(database)
    barrier = Barrier(2)

    def mutate(index):
        barrier.wait(timeout=5)
        try:
            return services.transport_mutate_work_plan(
                project_id,
                plan["id"],
                {"operation": "apply_template", "arguments": args},
                CommandContext(principal.id, f"apply-{index}", 0),
            )
        except ConflictError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(mutate, (1, 2)))
    assert sum(r is not None for r in results) == 1
    persisted = services.get_work_plan(project_id, plan["id"])
    assert persisted["revision"] == 1 and len(persisted["items"]) == 7
    assert len(persisted["template_applications"]) == 1
    assert len(services.events.load_stream(f"work-plan:{plan['id']}")) == 2
    assert services.get_work_template(project_id, template["id"], version=1) == template


def test_postgres_failed_plan_append_rolls_back_entire_materialization(database, monkeypatch):
    services, principal, project_id, template, plan, args = setup_template(database)
    original = PostgresEventStore.append

    def fail_after_insert(self, stream_id, **kwargs):
        result = original(self, stream_id, **kwargs)
        if stream_id == f"work-plan:{plan['id']}":
            raise RuntimeError("Injected after template plan event insertion")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(PostgresEventStore, "append", fail_after_insert)
        with pytest.raises(RuntimeError, match="Injected"):
            services.transport_mutate_work_plan(
                project_id,
                plan["id"],
                {"operation": "apply_template", "arguments": args},
                CommandContext(principal.id, "retryable", 0),
            )
    assert services.get_work_plan(project_id, plan["id"]) == plan
    assert len(services.events.load_stream(f"work-plan:{plan['id']}")) == 1
    changed = services.transport_mutate_work_plan(
        project_id,
        plan["id"],
        {"operation": "apply_template", "arguments": args},
        CommandContext(principal.id, "retryable", 0),
    )
    assert changed["revision"] == 1 and len(changed["items"]) == 7


def test_postgres_templates_are_versioned_and_cannot_be_hijacked_from_another_project(database):
    services, principal, project_id, template, plan, args = setup_template(database)
    other = project(services, principal, "Other")
    with pytest.raises(NotFoundError):
        services.transport_save_work_template(
            other["project_id"],
            definition(recipe(other["project_id"])),
            CommandContext(principal.id, "hijack", 1),
            template_id=template["id"],
        )
    updated = services.transport_save_work_template(
        project_id,
        definition(recipe(project_id)) | {"title": "V2"},
        CommandContext(principal.id, "v2", 1),
        template_id=template["id"],
    )
    assert updated["version"] == 2
    assert services.get_work_template(project_id, template["id"], version=1) == template
    assert services.list_work_templates(project_id) == [updated]
    with pytest.raises(ConflictError):
        services.transport_save_work_template(
            project_id,
            definition(recipe(project_id)),
            CommandContext(principal.id, "stale", 1),
            template_id=template["id"],
        )
