from dataclasses import replace
import threading
import hashlib
import sys
from time import monotonic, sleep
from urllib.error import HTTPError

import pytest

from kraken_agent.jobs import AgentJobState
from kraken_agent.runner import PluginRegistry
from kraken_agent.server_worker import ServerAgentWorker
from tests.test_agent_execution_safety import manifest


@pytest.mark.parametrize("enqueue_first", [False, True], ids=["while-staging", "while-queued"])
def test_revoked_lease_cancels_only_its_job_and_worker_can_execute_next(tmp_path, monkeypatch, enqueue_first):
    worker = ServerAgentWorker(
        server_url="http://127.0.0.1",
        token="test",
        data_dir=tmp_path,
        registry=PluginRegistry(),
        headers={"X-Kraken-Lease-Claim": "previous"},
    )
    worker.runner.environment["KRAKEN_ASSET_LEASE_CLAIM"] = "previous-env"
    revoked = threading.Event()
    entered = threading.Event()
    enqueue = worker.store.enqueue

    def staged_enqueue(item, **options):
        if enqueue_first:
            result = enqueue(item, **options)
        entered.set()
        assert revoked.wait(5)
        return result if enqueue_first else enqueue(item, **options)

    def request(method, path, payload=None, **kwargs):
        assert path.endswith("/heartbeat"), path
        assert entered.wait(5)
        raise HTTPError(path, 403, "superseded lease", None, None)

    original_heartbeat = worker._heartbeat

    def heartbeat(job_id, done, lost, deadline):
        original_heartbeat(job_id, done, lost, deadline)
        assert lost.is_set()
        revoked.set()

    monkeypatch.setattr("kraken_agent.telemetry.sample_host", lambda: {})
    monkeypatch.setattr(worker, "_request", request)
    monkeypatch.setattr(worker, "_heartbeat", heartbeat)
    monkeypatch.setattr(worker.store, "enqueue", staged_enqueue)
    job = manifest()
    response = worker.run_leased_job({"job_id": job.job_id, "lease_claim": "revoked", "manifest": job.to_dict()})
    assert response["imported"]["job"]["state"] == "cancelled"
    assert worker.store.get(job.job_id).state is AgentJobState.CANCELLED
    assert worker.headers["X-Kraken-Lease-Claim"] == "previous"
    assert worker.runner.environment["KRAKEN_ASSET_LEASE_CLAIM"] == "previous-env"

    # Even an unsupported next operation must reach the runner's failure path,
    # rather than remain queued because the preceding lease stopped the worker.
    enqueue(replace(job, job_id="next-job"))
    assert worker.runner.run_once(job_id="next-job") is True
    assert worker.store.get("next-job").state is AgentJobState.FAILED


@pytest.mark.parametrize("revision_race", [False, True], ids=["direct", "concurrent-revision"])
def test_network_loss_at_lease_deadline_terminates_actual_plugin_without_stopping_worker(
    tmp_path, monkeypatch, revision_race
):
    from tests.test_agent_execution_safety import staged_runner

    store, runner, workspace = staged_runner(
        tmp_path,
        "import os, pathlib, time; pathlib.Path(os.environ['KRAKEN_STAGING_ROOT'], 'started').touch(); time.sleep(30)",
    )
    worker = ServerAgentWorker(
        server_url="http://127.0.0.1", token="test", data_dir=tmp_path / "worker", registry=PluginRegistry()
    )
    worker.store, worker.runner = store, runner
    requests = []

    def unavailable(method, path, payload, *, timeout):
        requests.append(timeout)
        raise OSError("simulated connection loss")

    monkeypatch.setattr(worker, "_request", unavailable)
    monkeypatch.setattr("kraken_agent.telemetry.sample_host", lambda: {})
    done, lost = threading.Event(), threading.Event()
    if revision_race:
        from kraken_agent.jobs import JobStateError

        transition = store.transition
        raced = []

        def racing_transition(job_id, target, **kwargs):
            if target is AgentJobState.CANCELLED and not raced:
                raced.append(True)
                raise JobStateError("Simulated concurrent claim/state revision")
            return transition(job_id, target, **kwargs)

        monkeypatch.setattr(store, "transition", racing_transition)
    execution = threading.Thread(target=runner.run_once, kwargs={"cancel_event": lost}, daemon=True)
    execution.start()
    try:
        deadline = monotonic() + 5
        while not (workspace.path / "started").exists() and monotonic() < deadline:
            sleep(0.01)
        assert (workspace.path / "started").exists()
        worker._heartbeat("job-1", done, lost, monotonic() + 0.05)
        execution.join(5)
        assert lost.is_set() and not execution.is_alive()
        assert store.get("job-1").state is AgentJobState.CANCELLED
        assert requests and 0 < requests[0] <= 0.05
        assert not runner._stop.is_set()
    finally:
        done.set()
        runner.stop()
        execution.join(5)


def test_reclaimed_same_job_executes_in_fresh_workspace_and_keeps_old_attempt(tmp_path, monkeypatch):
    from kraken_agent.jobs import StagingWorkspace
    from kraken_agent.runner import PluginProcessSpec

    script = r"""
import hashlib, json, os
from pathlib import Path
from kraken_core.plugin_protocol import PluginResultManifest, PluginFrameOutput
job=json.loads(Path(os.environ['KRAKEN_JOB_MANIFEST']).read_text())
root=Path(os.environ['KRAKEN_STAGING_ROOT'])
assert (root/'inputs/1.png').read_bytes()==b'data'
assert not (root/'outputs/old-attempt.bin').exists()
data=b'fresh execution after lease reclaim'
(root/'outputs/probe.bin').write_bytes(data)
result=PluginResultManifest(job_id=job['job_id'], outcome='succeeded', plugin_id='recovery-probe',plugin_version='1',outputs=(
PluginFrameOutput(output_id='probe',frame_id='frame-1',relative_path='outputs/probe.bin',sha256=hashlib.sha256(data).hexdigest(),
media_type='application/octet-stream',role='diagnostic'),))
Path(os.environ['KRAKEN_RESULT_MANIFEST']).write_text(result.to_json())
"""
    registry = PluginRegistry(
        {"recovery.probe.v1": PluginProcessSpec("recovery.probe.v1", (sys.executable, "-c", script))}
    )
    worker = ServerAgentWorker(server_url="http://127.0.0.1", token="test", data_dir=tmp_path, registry=registry)
    retired = threading.Event()

    def heartbeat(job_id, done, lost, deadline):
        if worker.headers["X-Kraken-Lease-Claim"] == "first":
            worker._cancel_lost_lease(job_id, lost)
            retired.set()
        else:
            done.wait(5)

    enqueue = worker.store.enqueue

    def staged_enqueue(value, **options):
        assert retired.wait(5)
        return enqueue(value, **options)

    uploaded = []
    monkeypatch.setattr(worker, "_heartbeat", heartbeat)
    monkeypatch.setattr(worker.store, "enqueue", staged_enqueue)
    monkeypatch.setattr(worker, "_request", lambda *a, **k: {"accepted": True})
    monkeypatch.setattr(worker, "_upload", lambda url, path, **k: uploaded.append(path.read_bytes()))
    job = replace(manifest(), operation="recovery.probe.v1")
    assert (
        worker.run_leased_job({"job_id": job.job_id, "lease_claim": "first", "manifest": job.to_dict()})["imported"][
            "job"
        ]["state"]
        == "cancelled"
    )
    first = StagingWorkspace(tmp_path / "staging" / hashlib.sha256(b"first").hexdigest(), job.job_id)
    (first.path / "outputs/old-attempt.bin").write_bytes(b"stale output must be retained, not reused")
    second = StagingWorkspace(tmp_path / "staging" / hashlib.sha256(b"second").hexdigest(), job.job_id)
    second.create()
    (second.path / "inputs/1.png").write_bytes(b"data")
    assert worker.run_leased_job({"job_id": job.job_id, "lease_claim": "second", "manifest": job.to_dict()}) == {
        "accepted": True
    }
    assert uploaded == [b"fresh execution after lease reclaim"]
    assert (first.path / "outputs/old-attempt.bin").read_bytes() == b"stale output must be retained, not reused"
    history = worker.store.attempt_history(job.job_id)
    assert len(history) == 1 and history[0]["state"] == "cancelled"
    assert history[0]["attempt_key"] == hashlib.sha256(b"first").hexdigest()
