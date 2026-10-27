"""Real PostgreSQL global reservations and same-credential lease fencing."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import Barrier
from uuid import uuid4

import pytest
import sqlalchemy as sa

from kraken_manager.domain.pipeline import PipelineResources
from kraken_manager.domain.selection import FrameSelectionV1
from kraken_manager.domain.workflows import PluginInputV1, PluginJobManifestV1
from kraken_server.agent_auth import AgentIdentity
from kraken_server.agent_gateway import PostgresAgentGateway
from kraken_server.agent_resources import QueueResourceBudget

pytest_plugins = ["tests.test_admin_project_workflows"]


def manifest(*, parameters=None, project_id=None):
    return PluginJobManifestV1(
        job_id=str(uuid4()), project_id=project_id or str(uuid4()), layer_id=str(uuid4()),
        target_representation_id=str(uuid4()), selection=FrameSelectionV1.rectangle(1, 1, 1, 1),
        actor_principal_id=str(uuid4()), capability="frames.vectorize.v1",
        inputs=(PluginInputV1(str(uuid4()), str(uuid4()), "a" * 64, "inputs/1.png"),),
        parameters=parameters or {},
    )


@pytest.fixture
def queue(database):
    services = database[0]
    return PostgresAgentGateway(
        services.engine, None, services.uow_factory.blobs,
        resource_budget=QueueResourceBudget(4, PipelineResources(4, 1, 1024)),
    )


def agent():
    return AgentIdentity("same-machine-token", "worker", frozenset({"frames.vectorize.v1"}))


def declared(cpu=2, gpu=0, memory_mb=256):
    return {"execution_resources": dict(cpu=cpu, gpu=gpu, memory_mb=memory_mb)}


def claim(queue, identity=None, job_id=None):
    identity = identity or agent()
    lease = queue.lease(identity, job_id=job_id)
    assert lease is not None
    return lease, replace(identity, lease_claim=lease["lease_claim"])


def expire(queue, job_id):
    with queue.engine.begin() as connection:
        connection.execute(sa.update(queue.jobs).where(queue.jobs.c.job_id == job_id).values(
            lease_until=datetime.now(UTC) - timedelta(seconds=1),
        ))


def test_concurrent_admission_serializes_across_projects_and_process_gateways(queue):
    for _ in range(4):
        queue.submit(manifest(parameters=declared()))
    barrier = Barrier(4)

    def lease(_):
        independent = PostgresAgentGateway(queue.engine, None, queue.blobs, resource_budget=queue.admission.budget)
        barrier.wait()
        return independent.lease(agent())

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lease, range(4)))
    assert len([item for item in results if item]) == 2
    status = queue.resource_status()
    assert status["active_leases"] == 2
    assert status["declared_reserved"] == dict(cpu=4, gpu=0, memory_mb=512)
    assert status["scope"] == "global_server_queue"
    assert status["os_enforced"] is False
    assert status["per_host_capacity_enforced"] is False


@pytest.mark.parametrize("release", ["cancel", "complete", "fail", "expiry"])
def test_reservations_released_by_terminal_transition_or_crash_expiry(queue, release):
    first, next_job = manifest(), manifest(parameters=declared())
    queue.submit(first)  # A manual/legacy request conservatively reserves ALL capacity.
    queue.submit(next_job)
    leased, identity = claim(queue, job_id=first.job_id)
    assert queue.lease(agent(), job_id=next_job.job_id) is None
    assert queue.resource_status()["legacy_unknown_requests"] == 1
    if release == "cancel":
        queue.cancel(first.job_id)
    elif release == "expiry":
        expire(queue, first.job_id)
    else:
        queue.finish(first.job_id, identity, failed=release == "fail")
    assert queue.lease(agent(), job_id=next_job.job_id) is not None
    with pytest.raises(PermissionError):
        queue.heartbeat(leased["job_id"], identity)


def test_oversized_head_does_not_starve_a_fitting_job_after_100_rows(queue):
    for _ in range(103):
        queue.submit(manifest(parameters=declared(cpu=8)))
    fitting = manifest(parameters=declared())
    queue.submit(fitting)
    assert queue.lease(agent())["job_id"] == fitting.job_id
    assert queue.resource_status()["active_leases"] == 1


def test_expiry_reclaim_by_same_credential_rejects_every_old_generation_write(queue):
    job = manifest(parameters=declared())
    queue.submit(job)
    old, old_identity = claim(queue)
    expire(queue, job.job_id)
    current, current_identity = claim(queue)
    assert current["lease_claim"] != old["lease_claim"]
    called = []
    for operation in (
        lambda: queue.heartbeat(job.job_id, old_identity),
        lambda: queue.manifest(job.job_id, old_identity),
        lambda: queue.publish(job.job_id, old_identity, {"publication_id": "old"}),
        lambda: queue.upload_output(job.job_id, "old", old_identity, [b"old"], expected_sha256="a" * 64),
        lambda: queue.finish(job.job_id, old_identity, failed=False, import_result=lambda: called.append(True)),
        lambda: queue.finish(job.job_id, old_identity, failed=True),
    ):
        with pytest.raises(PermissionError, match="superseded"):
            operation()
    assert called == []
    with pytest.raises(PermissionError, match="update"):
        queue.heartbeat(job.job_id, agent())
    queue.heartbeat(job.job_id, current_identity)
    assert queue.publish(job.job_id, current_identity, {"publication_id": "new"})
    assert queue.finish(job.job_id, current_identity, failed=False, import_result=lambda: "winner") == "winner"
    assert queue.resource_status()["active_leases"] == 0


def test_partial_environment_is_rejected_and_unbounded_is_explicit(queue):
    with pytest.raises(ValueError, match="Declare all"):
        QueueResourceBudget.from_environment({"KRAKEN_QUEUE_CPU": "4"})
    assert QueueResourceBudget.from_environment({}) is None
    unbounded = PostgresAgentGateway(queue.engine, None, queue.blobs)
    status = unbounded.resource_status()
    assert status["state"] == "unenforced"
    assert status["admission_ready"] is False
    assert status["limits"] is None
    assert status["declared_reserved"] is None


def test_invalid_resource_request_rejected_before_enqueue(queue):
    with pytest.raises(ValueError, match="exactly"):
        queue.submit(manifest(parameters={"execution_resources": {"cpu": 1}}))
    assert queue.resource_status()["active_leases"] == 0


def test_http_claim_fencing_with_real_postgres_and_agent_credential(queue, database, monkeypatch):
    from fastapi.testclient import TestClient
    from kraken_server.app import create_app
    from kraken_server.agent_auth import PostgresAgentTokenStore

    services = database[0]
    tokens = PostgresAgentTokenStore(queue.engine)
    _, token = tokens.create("worker", agent().capabilities)
    job = manifest(parameters=declared())
    queue.submit(job)
    calls = []
    monkeypatch.setattr(services, "import_agent_result", lambda *args: calls.append("new") or {"accepted": True})
    client = TestClient(create_app(services=services, session_resolver=lambda _token: None,
                                  agent_token_store=tokens, agent_gateway=queue))
    authorization = {"Authorization": f"Bearer {token}"}
    first = client.post("/api/v1/agent/lease", headers=authorization, json={}).json()["job"]
    expire(queue, job.job_id)
    current = client.post("/api/v1/agent/lease", headers=authorization, json={}).json()["job"]
    base = f"/api/v1/agent/jobs/{job.job_id}"
    assert client.post(base + "/heartbeat", headers=authorization, json={}).status_code == 403
    stale = {**authorization, "X-Kraken-Lease-Claim": first["lease_claim"]}
    for path in ("/heartbeat", "/publications", "/complete", "/fail"):
        response = client.post(base + path, headers=stale, json={})
        assert response.status_code == 403, response.text
    assert calls == []
    headers = {**authorization, "X-Kraken-Lease-Claim": current["lease_claim"]}
    assert client.post(base + "/heartbeat", headers=headers, json={}).status_code == 200
    response = client.post(base + "/complete", headers=headers)
    assert response.status_code == 200, response.text
    assert calls == ["new"]


def test_latest_telemetry_is_generation_fenced_and_survives_completion(queue):
    from kraken_agent.telemetry import sample_host
    from types import SimpleNamespace
    sample = sample_host(
        psutil_module=SimpleNamespace(cpu_percent=lambda **k: 12,
                                      virtual_memory=lambda: SimpleNamespace(total=1024, available=768)),
        run=lambda *a, **k: SimpleNamespace(stdout="GPU, Actual model, 25, 100, 1000, [N/A]"),
    )
    job = manifest(parameters=declared())
    queue.submit(job)
    _, old = claim(queue)
    queue.heartbeat(job.job_id, old, telemetry=sample)
    expire(queue, job.job_id)
    _, new = claim(queue)
    sample["cpu_percent"] = 34
    queue.heartbeat(job.job_id, new, telemetry=sample)
    with pytest.raises(PermissionError):
        queue.heartbeat(job.job_id, old, telemetry={**sample, "cpu_percent": 99})
    queue.finish(job.job_id, new, failed=False)
    report = queue.host_observations()
    assert len(report) == 1
    assert report[0]["sample"]["cpu_percent"] == 34
    assert report[0]["sample"]["scope"] == "whole_host"
    assert report[0]["stale"] is False
