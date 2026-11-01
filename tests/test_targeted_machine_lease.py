"""A one-shot acceptance worker must never claim unrelated queue entries."""
from uuid import uuid4

from fastapi.testclient import TestClient
import sqlalchemy as sa

from kraken_agent.runner import PluginRegistry
from kraken_agent.server_worker import ServerAgentWorker
from tests.test_server_stitch_workflow import request, setup_stitch

pytest_plugins = ["tests.test_admin_project_workflows"]


def test_targeted_machine_lease_filters_before_any_queue_mutation(database):
    service, _, pid, queue, app, _, token, payload = setup_stitch(database)
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(app) as client:
        first = request(client, pid, payload)["model"]["stitch_renders"][-1]["job_id"]
        second = request(client, pid, payload, revision=3, key="second")["model"]["stitch_renders"][-1]["job_id"]
        with queue.engine.connect() as c:
            before = [dict(r) for r in c.execute(sa.select(queue.jobs).order_by(queue.jobs.c.job_id)).mappings()]
        unknown = client.post(f"/api/v1/agent/jobs/{uuid4()}/lease", headers=headers,
            json={"capabilities": ["layer.stitch.v1"]})
        assert unknown.status_code == 200 and unknown.json() == {"job": None}
        invalid = client.post("/api/v1/agent/jobs/invalid/lease", headers=headers, json={})
        assert invalid.status_code == 422
        denied = client.post(f"/api/v1/agent/jobs/{second}/lease", headers=headers,
            json={"capabilities": ["karakal.analyze.v1"]})
        assert denied.status_code == 403
        with queue.engine.connect() as c:
            after = [dict(r) for r in c.execute(sa.select(queue.jobs).order_by(queue.jobs.c.job_id)).mappings()]
        assert after == before
        selected = client.post(f"/api/v1/agent/jobs/{second}/lease", headers=headers,
            json={"capabilities": ["layer.stitch.v1"]})
        assert selected.status_code == 200 and selected.json()["job"]["job_id"] == second
        with queue.engine.connect() as c:
            unchanged = dict(c.execute(sa.select(queue.jobs).where(queue.jobs.c.job_id == first)).mappings().one())
        assert unchanged == next(r for r in before if str(r["job_id"]) == first)


def test_worker_sends_explicit_target_without_changing_default_polling(tmp_path, monkeypatch):
    worker = ServerAgentWorker(server_url="http://127.0.0.1:8080", token="test-only",
        data_dir=tmp_path, registry=PluginRegistry({}))
    requests = []
    monkeypatch.setattr(worker, "_request", lambda method, path, payload: requests.append(path) or {"job": None})
    target = str(uuid4())
    assert worker.run_once(job_id=target) is False
    assert requests[-1] == f"/api/v1/agent/jobs/{target}/lease"
    assert worker.run_once() is False
    assert requests[-1] == "/api/v1/agent/lease"
