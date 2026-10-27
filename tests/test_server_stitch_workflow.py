"""Actual PG/HTTP/leased Agent/native mosaic, without model training."""
from datetime import UTC, datetime, timedelta
from dataclasses import replace
import socket
import sys
import threading
import time
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest
import sqlalchemy as sa
import uvicorn

from kraken_admin.local_admin import open_project_services
from kraken_agent.runner import PluginProcessSpec, PluginRegistry
from kraken_agent.server_worker import ServerAgentWorker
from kraken_manager.domain.artifacts import deterministic_frame_series_id
from kraken_manager.domain.multilayer import MultilayerModel
from kraken_manager.infrastructure.mosaic import MosaicReader, STITCH_OPERATION
from kraken_server.agent_auth import PostgresAgentTokenStore
from kraken_server.agent_gateway import PostgresAgentGateway
from kraken_server.app import SessionPrincipal, create_app
from kraken_server.services import CommandContext
from tests.test_admin_project_workflows import project
from tests.test_electrical_graph import png

pytest_plugins = ["tests.test_admin_project_workflows"]


def setup_stitch(database, *, corrupt=False, tile_size=4):
    service, _, owner, _ = database
    pid = project(service, owner, "Controlled stitch workflow")["project_id"]
    layer = service.create_layer(pid, {"name": "M1", "type": "metal", "order": 0},
        CommandContext(owner.id, "layer", service.get_project(pid)["revision"]))
    lid = layer["layer_id"]
    rid = service.create_representation(pid, lid, {"name": "Inputs", "kind": "image"},
        CommandContext(owner.id, "images", layer["revision"]))["representation_id"]
    tiles, coordinates = [], {}
    for i in range(2):
        frame = service.projections.get_project(pid).frame_id_at(i+1, 1)
        series = service.create_artifact_series(pid, {"name": f"tile-{i}", "scope": "frame_representation",
            "layer_id": lid, "representation_id": rid, "frame_id": frame,
            "series_id": deterministic_frame_series_id(rid, frame)}, CommandContext(owner.id, f"series-{i}", None))
        version = service.add_managed_artifact_version(pid, series["id"], {"filename": f"tile-{i}.png", "media_type": "image/png"},
            (b"corrupted image fixture" if corrupt and i else png([[i*100]*tile_size]*tile_size),),
            CommandContext(owner.id, f"version-{i}", service.artifact_stream_revision(pid, series["id"])))
        tiles.append({"frame_id": frame, "artifact_version_id": version["id"], "width": tile_size, "height": tile_size,
                      "pixel_size_m": [1e-6, 1e-6], "y_axis": "down", "catalog_xy": [i+1, 1]})
        coordinates[str(frame)] = [i+1, 1]
    tiles[1].update(declared_affine=[1, 0, 2e-6, 0, 1, 0], note="controlled functional layout, not real stage metadata")
    series = service.create_artifact_series(pid, {"name": "Stage evidence", "scope": "layer_attachment", "layer_id": lid},
        CommandContext(owner.id, "evidence-series", None))
    evidence = service.add_managed_artifact_version(pid, series["id"], {"filename": "observations.json", "media_type": "application/json"},
        (b'{"fixture":"functional"}',), CommandContext(owner.id, "evidence", service.artifact_stream_revision(pid, series["id"])))
    service.transport_mutate_multilayer_model(pid, {"operation": "set_stack", "arguments": {"layers": [
        {"layer_id": lid, "layer_type": "metal", "z_m": 0, "thickness_m": 0}]}}, CommandContext(owner.id, "stack", 0))
    geometry = service.transport_mutate_multilayer_model(pid, {"operation": "plan_layer_stitch", "arguments": {
        "layer_id": lid, "plan_version_id": evidence["id"], "reference_frame_id": tiles[0]["frame_id"], "tiles": tiles,
        "max_error_m": 1e-12}}, CommandContext(owner.id, "geometry", 1))
    stitch = MultilayerModel.from_dict(geometry).stitches[-1]
    tokens = PostgresAgentTokenStore(service.engine)
    queue = PostgresAgentGateway(service.engine, tokens, service.uow_factory.blobs)
    service.agent_gateway = queue
    identity, token = tokens.create("stitch-worker", frozenset({STITCH_OPERATION}))
    app = create_app(services=service, agent_token_store=tokens, agent_gateway=queue,
        session_resolver=lambda value: SessionPrincipal(str(owner.id), "local", value) if value == "owner" else None)
    payload = {"operation": "request_layer_stitch", "arguments": {"stitch_id": stitch.id, "stitch_revision": stitch.revision,
        "mode": "gray", "block_size": 2, "frame_coordinates": coordinates}}
    return service, owner, pid, queue, app, identity, token, payload


def request(client, pid, payload, *, revision=2, key="request"):
    response = client.post(f"/api/v1/projects/{pid}/multilayer/commands", json=payload,
        headers={"Authorization": "Bearer owner", "Idempotency-Key": key, "If-Match": str(revision)})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("corrupt", [False, True])
def test_real_remote_stitch_native_worker_and_atomic_import(database, tmp_path, corrupt):
    service, _, pid, queue, app, _, token, payload = setup_stitch(database, corrupt=corrupt)
    with TestClient(app) as client:
        first = request(client, pid, payload)
        job = first["model"]["stitch_renders"][-1]
        assert job["state"] == "requested"
        assert request(client, pid, payload, revision=0) == first
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
        assert server.started
        assert worker.run_once()
        snapshot = service.get_multilayer_model(pid)
        render = MultilayerModel.from_dict(snapshot).stitch_renders[-1]
        assert render.state == ("failed" if corrupt else "imported"), snapshot
        if corrupt:
            assert render.error and render.output is None
            with queue.engine.begin() as connection:
                connection.execute(sa.update(queue.jobs).where(queue.jobs.c.job_id == render.job_id).values(
                    state="leased", lease_owner="retired-failure", lease_claim="retired-failure-claim",
                    lease_until=datetime.now(UTC)-timedelta(seconds=1)))
            service.run_stitch_cycle()
            assert service.get_multilayer_model(pid) == snapshot
        else:
            version = service.projections.get_artifact_version(render.output.version_id)
            series = service.projections.get_artifact_series(version.series_id)
            assert series.scope.value == "layer_attachment" and series.frame_id is None
            assert len(version.input_version_ids) == 3
            with MosaicReader(service.uow_factory.blobs.path_for_read(version.blob)) as reader:
                reader.validate(expected_job_id=render.job_id)
                assert reader.metadata["render"]["block_count"] == 6
                assert reader.read_block(2, 0).coverage.max() == 2
            # Simulate the separate queue transaction being lost after the
            # canonical receipt committed. Recovery must not recompute or
            # accept a new publication for this already accepted job.
            with queue.engine.begin() as connection:
                connection.execute(sa.update(queue.jobs).where(queue.jobs.c.job_id == render.job_id).values(
                    state="leased", lease_owner="retired-transport", lease_claim="retired-claim",
                    lease_until=datetime.now(UTC)-timedelta(seconds=1)))
            service.run_stitch_cycle()
            assert service.get_multilayer_model(pid) == snapshot
        reopened = open_project_services(service.engine.url.render_as_string(hide_password=False), service.uow_factory.blobs.root,
                                         engine=service.engine)
        assert reopened.get_multilayer_model(pid) == snapshot
        with queue.engine.connect() as connection:
            assert connection.execute(sa.select(queue.jobs.c.state).where(queue.jobs.c.job_id == render.job_id)).scalar_one() == (
                "cancelled" if corrupt else "completed")
    finally:
        worker.stop()
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


def test_stitch_lease_fences_inputs_publication_and_completion_after_reclaim_or_cancel(database):
    service, _, pid, queue, app, identity, token, payload = setup_stitch(database)
    with TestClient(app) as client:
        first = request(client, pid, payload)
        job_id = first["model"]["stitch_renders"][-1]["job_id"]
        lease = queue.lease(identity, seconds=15)
        assert lease["job_id"] == job_id
        headers = {"Authorization": f"Bearer {token}", "X-Kraken-Lease-Claim": lease["lease_claim"]}
        asset = lease["manifest"]["inputs"][0]
        base = f"/api/v1/agent/jobs/{job_id}"
        assert client.get(base + "/inputs/" + asset["asset_id"], headers=headers).status_code == 200
        assert client.get(base + "/inputs/" + str(uuid4()), headers=headers).status_code == 403
        with queue.engine.begin() as connection:
            connection.execute(sa.update(queue.jobs).where(queue.jobs.c.job_id == job_id)
                .values(lease_until=datetime.now(UTC)-timedelta(seconds=1)))
        newer = queue.lease(identity, seconds=15)
        assert newer["lease_claim"] != lease["lease_claim"]
        for suffix, body in (("/publications", {}), ("/complete", {}), ("/fail", {"error": "old claim"})):
            assert client.post(base + suffix, json=body, headers=headers).status_code == 403
        assert client.get(base + "/inputs/" + asset["asset_id"], headers=headers).status_code == 403
        current_headers = headers | {"X-Kraken-Lease-Claim": newer["lease_claim"]}
        cancelled = request(client, pid, {"operation": "cancel_layer_stitch", "arguments": {"job_id": job_id}},
                            revision=3, key="cancel")
        assert cancelled["model"]["stitch_renders"][-1]["state"] == "cancelled"
        assert client.post(base + "/complete", json={}, headers=current_headers).status_code == 403
        assert service.get_multilayer_model(pid)["model"]["stitch_renders"][-1]["output"] is None


def test_committed_request_recovers_after_enqueue_outage_and_cancel_before_enqueue(database, monkeypatch):
    service, _, pid, queue, app, identity, _, payload = setup_stitch(database)
    submit = queue.submit_stitch

    def offline(*_):
        raise OSError("controlled queue outage")

    monkeypatch.setattr(queue, "submit_stitch", offline)
    with TestClient(app) as client:
        first = request(client, pid, payload)
        job_id = first["model"]["stitch_renders"][-1]["job_id"]
        with queue.engine.connect() as connection:
            assert connection.execute(sa.select(queue.jobs.c.state).where(queue.jobs.c.job_id == job_id)).scalar_one_or_none() is None
        monkeypatch.setattr(queue, "submit_stitch", submit)
        service.run_stitch_cycle()
        assert queue.lease(identity, job_id=job_id) is not None
        assert service.get_multilayer_model(pid) == first
        request(client, pid, {"operation": "cancel_layer_stitch", "arguments": {"job_id": job_id}}, revision=3, key="cancel")
        with queue.engine.connect() as connection:
            assert connection.execute(sa.select(queue.jobs.c.state).where(queue.jobs.c.job_id == job_id)).scalar_one() == "cancelled"


def test_one_pending_enqueue_does_not_starve_later_requests(database, monkeypatch):
    service, _, pid, queue, app, _, _, payload = setup_stitch(database)
    submit = queue.submit_stitch
    blocked = []

    def selective(manifest):
        if not blocked:
            blocked.append(manifest.job_id)
        if manifest.job_id == blocked[0]:
            raise OSError("controlled failure of one queue intent")
        submit(manifest)

    monkeypatch.setattr(queue, "submit_stitch", selective)
    with TestClient(app) as client:
        first = request(client, pid, payload)
        second = request(client, pid, payload, revision=3, key="later-request")
        first_id = first["model"]["stitch_renders"][-1]["job_id"]
        second_id = second["model"]["stitch_renders"][-1]["job_id"]
        with queue.engine.connect() as connection:
            assert connection.execute(sa.select(queue.jobs.c.state).where(queue.jobs.c.job_id == first_id)).scalar_one_or_none() is None
            assert connection.execute(sa.select(queue.jobs.c.state).where(queue.jobs.c.job_id == second_id)).scalar_one() == "queued"


def test_cancellation_committed_during_enqueue_wins_the_race(database, monkeypatch):
    service, owner, pid, queue, app, identity, _, payload = setup_stitch(database)
    submit = queue.submit_stitch

    def cancel_in_enqueue(manifest):
        submit(manifest)
        service.transport_mutate_multilayer_model(pid, {"operation": "cancel_layer_stitch",
            "arguments": {"job_id": manifest.job_id}}, CommandContext(owner.id, "cancel-race", 3))

    monkeypatch.setattr(queue, "submit_stitch", cancel_in_enqueue)
    with TestClient(app) as client:
        request(client, pid, payload)
    state = service.get_multilayer_model(pid)["model"]["stitch_renders"][-1]
    assert state["state"] == "cancelled" and state["output"] is None
    assert queue.lease(identity, job_id=state["job_id"]) is None


def test_initiator_deactivation_stops_queued_work_and_records_scheduler_rejection(database):
    service, owner, pid, queue, app, identity, _, payload = setup_stitch(database)
    with TestClient(app) as client:
        first = request(client, pid, payload)
    job_id = first["model"]["stitch_renders"][-1]["job_id"]
    service.identities.save(replace(owner, active=False))
    service.run_stitch_cycle()
    state = service.get_multilayer_model(pid)["model"]["stitch_renders"][-1]
    assert state["state"] == "failed" and state["error"].startswith("Authorization lost:") and state["output"] is None
    assert queue.lease(identity, job_id=job_id) is None
    event = service.events.load_stream(f"multilayer:{pid}")[-1]
    assert event.payload["operation"] == "reject_layer_stitch_authorization"
    assert event.program.name == "Kraken Server scheduler"
    assert event.actor.principal_id == owner.id
