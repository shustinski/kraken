"""Failure boundaries of the real layer adapter, not vectorization quality."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import sqlalchemy as sa

from kraken_manager.domain.multilayer import MultilayerModel
from kraken_manager.infrastructure.postgres.event_store import PostgresEventStore, PostgresRevisionConflict
from tests.test_pipeline_stitch_postgres import pipeline
from tests.test_pipeline_stitch_runtime import define
from tests.test_stitch_local_workflow import computing as computing
from tests.test_stitch_plan_workflow import geometry as geometry

pytest_plugins = ["tests.test_admin_project_workflows"]


def test_quality_rejection_after_native_finish_then_explicit_retry_gets_new_attempt(computing):
    local, session, project, versions, _, _, gateway, runner, _ = computing
    op, _, args = define(computing)
    state = local.tick_layer_pipeline(**args, stitch_gateway=gateway)
    first = state["nodes"][op.id]["attempts"][-1]["job_id"]
    assert runner.run_once()
    issue = local.create_quality_issue(project_id=project.id, principal=session.principal, idempotency_key="late-quality",
        arguments=dict(title="Source review required", description="Functional rejection at import boundary",
            severity="critical", scope={"layer_ids": [args["layer_id"]]}, source_version_ids=[str(versions[0].id)]))
    state = local.tick_layer_pipeline(**args, stitch_gateway=gateway)
    assert state["nodes"][op.id]["status"] == "failed"
    render = next(r for r in MultilayerModel.from_dict(local.get_multilayer_model(project.id)).stitch_renders if r.job_id == first)
    assert render.state == "failed" and render.output is None and issue.id in render.error
    local.mutate_quality_issue(project_id=project.id, issue_id=issue.id, principal=session.principal,
        expected_revision=issue.revision, operation="exception", arguments={"note": "Controlled functional run only"}, idempotency_key="exception")
    local.command_layer_pipeline(**args, operation="retry", node_id=op.id, expected_revision=state["revision"], idempotency_key="retry")
    state = local.tick_layer_pipeline(**args, stitch_gateway=gateway)
    attempts = state["nodes"][op.id]["attempts"]
    assert len(attempts) == 2 and attempts[-1]["job_id"] != first
    assert runner.run_once()
    state = local.tick_layer_pipeline(**args, stitch_gateway=gateway)
    assert state["status"] == "completed"
    assert len(state["nodes"][op.id]["attempts"]) == 2


def test_postgres_layer_concurrent_reservation_has_one_native_request(database, monkeypatch):
    service, actor, pid, lid, op, queue, _, _, _, initial = pipeline(database)
    barrier = Barrier(2)
    append = PostgresEventStore.append

    def synchronized(self, stream_id, *, expected_revision, events):
        if stream_id == f"layer-pipeline:{lid}" and expected_revision == initial["revision"]:
            barrier.wait(timeout=10)
        return append(self, stream_id, expected_revision=expected_revision, events=events)

    monkeypatch.setattr(PostgresEventStore, "append", synchronized)

    def tick(_):
        try:
            return service._layer_runtime().tick(pid, lid, principal=actor)
        except PostgresRevisionConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(tick, (1, 2)))
    assert sum(value is not None for value in results) == 1
    state = service.get_layer_pipeline(pid, lid)
    attempts = state["nodes"][op.id]["attempts"]
    assert len(attempts) == 1
    renders = MultilayerModel.from_dict(service.get_multilayer_model(pid)).stitch_renders
    assert len(renders) == 1 and renders[0].job_id == attempts[0]["job_id"]
    with queue.engine.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(queue.jobs)).scalar_one() == 1


def test_postgres_fault_after_physical_append_rolls_back_both_streams(database, monkeypatch):
    service, actor, pid, lid, _, queue, _, _, _, initial = pipeline(database)
    before = service.get_multilayer_model(pid)
    append = PostgresEventStore.append

    def fail(self, stream_id, *, expected_revision, events):
        result = append(self, stream_id, expected_revision=expected_revision, events=events)
        if events[-1].payload.get("operation") == "pipeline_stitch_requested":
            raise OSError("controlled PG append boundary")
        return result

    monkeypatch.setattr(PostgresEventStore, "append", fail)
    with pytest.raises(OSError, match="PG append boundary"):
        service._layer_runtime().tick(pid, lid, principal=actor)
    assert service.get_layer_pipeline(pid, lid) == initial
    assert service.get_multilayer_model(pid) == before
    with queue.engine.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(queue.jobs)).scalar_one() == 0
