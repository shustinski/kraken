"""Real PostgreSQL tests; each fixture owns a UUID schema and temporary files."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from kraken_manager.domain.identity import Performer, Principal
from kraken_manager.domain.selection import FrameSelectionV1
from kraken_manager.domain.work_orders import WorkKind, WorkOrder, WorkPlacement
from kraken_manager.infrastructure.filesystem._codec import encode_model
from kraken_manager.infrastructure.postgres.performer_store import PostgresPerformerStore
from kraken_server.services import CommandContext, ConflictError, ForbiddenError
from tests.test_admin_project_workflows import project

pytest_plugins = ["tests.test_admin_project_workflows"]


def test_postgres_plan_persists_replays_and_rejects_concurrent_stale_update(database):
    services, _accounts, owner, _config = database
    project_id = project(services, owner)["project_id"]
    created = services.transport_create_work_plan(project_id, {"title": "Restore"}, CommandContext(owner.id, "plan", None))
    assert services.transport_create_work_plan(project_id, {"title": "Restore"}, CommandContext(owner.id, "plan", None)) == created
    barrier = Barrier(2)

    def mutate(index):
        barrier.wait(timeout=5)
        try:
            return services.transport_mutate_work_plan(
                project_id, created["id"], {"operation": "add_item", "arguments": {"item": {"title": f"Stage {index}"}}},
                CommandContext(owner.id, f"item-{index}", 0),
            )
        except ConflictError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(mutate, (1, 2)))
    assert sum(item is not None for item in results) == 1
    persisted = services.get_work_plan(project_id, created["id"])
    assert persisted["revision"] == 1 and len(persisted["items"]) == 1
    assert services.list_work_plans(project_id) == [persisted]
    assert len(services.events.load_stream(f"work-plan:{created['id']}")) == 2


def test_postgres_work_order_metadata_and_event_commit_with_rbac_and_cas(database, tmp_path):
    services, _accounts, owner, _config = database
    created_project = project(services, owner)
    identifier = created_project["project_id"]
    layer = services.create_layer(identifier, {"name": "M1", "type": "metal", "order": 0}, CommandContext(owner.id, "layer", created_project["revision"]))
    services.performer_store = PostgresPerformerStore(services.engine)
    performer = services.performer_store.create(Performer.create(name="Engineer", color="#446688"))
    order = WorkOrder.create(
        project_id=identifier, layer_id=layer["layer_id"], kind=WorkKind.STITCH,
        placement=WorkPlacement.EXTERNAL, assignee_id=performer.id, created_by=owner.id,
        destination=str(tmp_path / "external"), frame_count=1, selection=FrameSelectionV1.rectangle(1, 1, 1, 1),
    )
    created = services.issue_work_order(identifier, {"record": encode_model(order)}, CommandContext(owner.id, "order", None))
    assert created["revision"] == 0
    outsider = Principal.local(subject="outsider", display_name="Outsider")
    services.identities.save(outsider)
    with pytest.raises(ForbiddenError):
        services.cancel_work_order(identifier, order.id, CommandContext(outsider.id, "forbidden", 0))
    assert services.projections.get_work_order(order.id).state.value == "issued"
    cancelled = services.cancel_work_order(identifier, order.id, CommandContext(owner.id, "cancel", 0))
    assert cancelled["state"] == "cancelled" and cancelled["revision"] == 1
    assert services.cancel_work_order(identifier, order.id, CommandContext(owner.id, "cancel", 0)) == cancelled
    with pytest.raises(ConflictError):
        services.cancel_work_order(identifier, order.id, CommandContext(owner.id, "stale", 0))
    assert len(services.events.load_stream(f"work-order:{order.id}")) == 2
