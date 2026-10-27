"""Native layer pipeline on PostgreSQL and the actual HTTP Agent transport."""
import socket
import sys
import threading
import time
from uuid import uuid4

import pytest
import sqlalchemy as sa
import uvicorn

from kraken_agent.runner import PluginProcessSpec, PluginRegistry
from kraken_agent.server_worker import ServerAgentWorker
from kraken_manager.domain.multilayer import MultilayerModel
from kraken_manager.domain.pipeline import LayerPipelineDefinition, PipelinePolicy, PipelineResources
from kraken_manager.domain.pipeline_layer import PipelineLayerOperation
from kraken_manager.infrastructure.mosaic import MosaicReader, STITCH_OPERATION
from kraken_server.services import CommandContext
from tests.test_server_stitch_workflow import setup_stitch

pytest_plugins = ["tests.test_admin_project_workflows"]


def pipeline(database):
    service, actor, pid, queue, app, identity, token, _ = setup_stitch(database)
    stitch = MultilayerModel.from_dict(service.get_multilayer_model(pid)).stitches[-1]
    operation = PipelineLayerOperation(str(uuid4()), "Native layer", STITCH_OPERATION, "1.0.0", stitch.id, stitch.revision,
        parameters={"mode": "gray", "block_size": 2}, resources=PipelineResources(cpu=2, gpu=0, memory_mb=640))
    definition = LayerPipelineDefinition(pid, stitch.layer_id, (operation,), policy=PipelinePolicy(capacity=operation.resources))
    runtime = service._layer_runtime()
    state = runtime.define(pid, stitch.layer_id, definition.to_dict(), principal=actor, expected_revision=0, idempotency_key="define")
    state = runtime.command(pid, stitch.layer_id, "start", principal=actor, expected_revision=state["revision"], idempotency_key="start")
    return service, actor, pid, stitch.layer_id, operation, queue, app, identity, token, state


def test_native_layer_pipeline_actual_server_agent_and_reopen(database, tmp_path):
    service, actor, pid, lid, op, queue, app, _, token, _ = pipeline(database)
    state = service._layer_runtime().tick(pid, lid, principal=actor)
    attempt = state["nodes"][op.id]["attempts"][-1]
    with queue.engine.connect() as connection:
        payload = connection.execute(sa.select(queue.jobs.c.payload).where(queue.jobs.c.job_id == attempt["job_id"])).scalar_one()
    assert payload["parameters"]["execution_resources"] == op.resources.to_dict()
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    worker = ServerAgentWorker(server_url=f"http://127.0.0.1:{sock.getsockname()[1]}", token=token,
        data_dir=tmp_path / "agent", registry=PluginRegistry({STITCH_OPERATION: PluginProcessSpec(STITCH_OPERATION,
            (sys.executable, "-m", "kraken_manager.infrastructure.stitch_worker"))}))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(.01)
        assert server.started and worker.run_once()
        state = service._layer_runtime().tick(pid, lid, principal=actor)
        assert state["status"] == "completed", state
        finished = state["nodes"][op.id]["attempts"][-1]
        assert finished["job_id"] == attempt["job_id"] and len(finished["output_version_ids"]) == 1
        version = service.projections.get_artifact_version(finished["output_version_ids"][0])
        with MosaicReader(service.uow_factory.blobs.path_for_read(version.blob)) as reader:
            reader.validate(expected_job_id=attempt["job_id"])
            assert reader.read_block(2, 0).coverage.max() == 2
        assert service._layer_runtime().get(pid, lid) == state
    finally:
        worker.stop()
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


@pytest.mark.parametrize("command", ["pause", "cancel"])
def test_global_stitch_recovery_obeys_pipeline_pause_and_cancel(database, monkeypatch, command):
    service, actor, pid, lid, op, queue, _, identity, _, _ = pipeline(database)
    submit = queue.submit_stitch
    monkeypatch.setattr(queue, "submit_stitch", lambda *_: (_ for _ in ()).throw(OSError("controlled enqueue outage")))
    state = service._layer_runtime().tick(pid, lid, principal=actor)
    job = state["nodes"][op.id]["attempts"][-1]["job_id"]
    service.transport_layer_pipeline(pid, lid, {"operation": command}, CommandContext(actor.id, command, state["revision"]))
    monkeypatch.setattr(queue, "submit_stitch", submit)
    service.run_stitch_cycle()
    assert queue.lease(identity, job_id=job) is None
    if command == "pause":
        state = service.get_layer_pipeline(pid, lid)
        service.transport_layer_pipeline(pid, lid, {"operation": "resume"}, CommandContext(actor.id, "resume", state["revision"]))
        service.run_stitch_cycle()
        assert queue.lease(identity, job_id=job)["job_id"] == job
        assert len(service.get_layer_pipeline(pid, lid)["nodes"][op.id]["attempts"]) == 1
    else:
        render = next(r for r in MultilayerModel.from_dict(service.get_multilayer_model(pid)).stitch_renders if r.job_id == job)
        assert render.state == "cancelled" and render.output is None
