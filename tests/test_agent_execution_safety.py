from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from kraken_agent.jobs import AgentJobState, DurableJobStore, JobStateError, StagingWorkspace
from kraken_agent.runner import PluginProcessSpec, PluginRegistry, SubprocessPluginRunner
from kraken_agent.service import AgentControlServer
from kraken_core.plugin_protocol import (
    PluginFrameInput, PluginFrameOutput, PluginJobManifest, PluginResultManifest, PluginResultPublicationV2,
)


def manifest(job_id="job-1"):
    return PluginJobManifest(
        job_id=job_id, operation="frames.vectorize.v1", project_id="project-1",
        layer_id="layer-1", actor_id="actor-1", target_representation_id="representation-1",
        inputs=(PluginFrameInput("frame-1", 1, 1, "version-1", hashlib.sha256(b"data").hexdigest(),
                                 "image/png", "inputs/1.png"),),
    )


def publication(*, outcome="succeeded"):
    return PluginResultPublicationV2(
        job_id="job-1", publication_id="publication-1", sequence=1,
        plugin_id="test", plugin_version="1", outputs=(), final=True, outcome=outcome,
    )


def running(store):
    store.enqueue(manifest())
    job = store.claim_next()
    return store.transition(job.job_id, AgentJobState.RUNNING, expected_revision=job.revision)


def test_claim_is_fifo_and_atomic_across_store_instances(tmp_path):
    database = tmp_path / "jobs.sqlite3"
    store = DurableJobStore(database)
    store.enqueue(manifest("first"))
    store.enqueue(manifest("second"))
    # Listing remains newest first for the UI; execution must use creation order.
    assert store.claim_next().job_id == "first"
    stores = [DurableJobStore(database) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        claims = list(executor.map(lambda candidate: candidate.claim_next(), stores))
    assert [job.job_id for job in claims if job is not None] == ["second"]


def test_claim_specific_lease_leaves_other_queued_jobs_untouched(tmp_path):
    store = DurableJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue(manifest("older"))
    store.enqueue(manifest("leased"))
    assert store.claim_next(job_id="leased").job_id == "leased"
    assert store.get("older").state is AgentJobState.QUEUED
    assert store.claim_next(job_id="leased") is None


def test_stopped_runner_does_not_claim_or_launch_jobs(tmp_path):
    store = DurableJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue(manifest())
    runner = SubprocessPluginRunner(store, tmp_path / "staging", PluginRegistry())
    runner.stop()
    assert runner.run_once() is False
    assert store.get("job-1").state is AgentJobState.QUEUED


def staged_runner(tmp_path, script):
    store = DurableJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue(manifest())
    source = tmp_path / "input.png"
    source.write_bytes(b"data")
    workspace = StagingWorkspace(tmp_path / "staging", "job-1")
    workspace.create()
    workspace.stage_file(source, "inputs/1.png", expected_sha256=hashlib.sha256(b"data").hexdigest())
    registry = PluginRegistry({
        "frames.vectorize.v1": PluginProcessSpec("frames.vectorize.v1", (sys.executable, "-c", script)),
    })
    return store, SubprocessPluginRunner(store, tmp_path / "staging", registry), workspace


def test_stopping_active_subprocess_marks_recovery_required(tmp_path):
    store, runner, workspace = staged_runner(
        tmp_path,
        "import os, pathlib, time; "
        "pathlib.Path(os.environ['KRAKEN_STAGING_ROOT'], 'started').write_text('ready'); time.sleep(60)",
    )
    thread = threading.Thread(target=runner.run_once, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not (workspace.path / "started").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (workspace.path / "started").exists()
    finally:
        runner.stop()
        thread.join(timeout=15)
    assert not thread.is_alive()
    job = store.get("job-1")
    assert job.state is AgentJobState.RECOVERY_REQUIRED
    assert job.result is None


@pytest.mark.parametrize("cancel_after_exit", [False, True])
def test_subprocess_final_result_is_not_persisted_after_cancellation(tmp_path, monkeypatch, cancel_after_exit):
    result = PluginResultManifest(
        job_id="job-1", outcome="failed", plugin_id="test", plugin_version="1", outputs=(),
    )
    script = (
        "import os, pathlib; "
        f"pathlib.Path(os.environ['KRAKEN_RESULT_MANIFEST']).write_text({result.to_json()!r}, encoding='utf-8')"
    )
    store, runner, _workspace = staged_runner(tmp_path, script)
    if cancel_after_exit:
        def cancel_before_persistence(_workspace, _result, _manifest):
            job = store.get("job-1")
            store.transition(job.job_id, AgentJobState.CANCELLED, expected_revision=job.revision)
        monkeypatch.setattr(StagingWorkspace, "verify_result", cancel_before_persistence)
    assert runner.run_once()
    job = store.get("job-1")
    assert job.state is (AgentJobState.CANCELLED if cancel_after_exit else AgentJobState.FAILED)
    if cancel_after_exit:
        assert job.result is None
    else:
        assert job.result == result


def test_successful_subprocess_result_waits_for_import(tmp_path):
    result = PluginResultManifest(
        job_id="job-1", outcome="succeeded", plugin_id="test", plugin_version="1",
        outputs=(PluginFrameOutput(
            "vector-1", "frame-1", "outputs/1.cif", hashlib.sha256(b"CIF").hexdigest(),
            "application/x-cif", "vector",
        ),),
    )
    script = (
        "import os, pathlib; "
        "pathlib.Path(os.environ['KRAKEN_STAGING_ROOT'], 'outputs', '1.cif').write_bytes(b'CIF'); "
        f"pathlib.Path(os.environ['KRAKEN_RESULT_MANIFEST']).write_text({result.to_json()!r}, encoding='utf-8')"
    )
    store, runner, _workspace = staged_runner(tmp_path, script)
    assert runner.run_once()
    job = store.get("job-1")
    assert job.state is AgentJobState.IMPORTING
    assert job.result == result


def test_cancelled_job_rejects_late_result_without_recording_callback(tmp_path):
    store = DurableJobStore(tmp_path / "jobs.sqlite3")
    job = running(store)
    job = store.transition(job.job_id, AgentJobState.CANCELLED, expected_revision=job.revision)
    with pytest.raises(JobStateError, match="cannot publish"):
        store.record_result(publication(), callback_key="result", expected_revision=job.revision)
    assert store.get(job.job_id).result is None
    assert store.list_publications(job.job_id) == ()


def test_result_and_final_state_are_committed_together_and_retry_is_idempotent(tmp_path):
    store = DurableJobStore(tmp_path / "jobs.sqlite3")
    job = running(store)
    completed, duplicate = store.record_result(
        publication(), callback_key="result", expected_revision=job.revision,
        target_state=AgentJobState.IMPORTING, allowed_states=(AgentJobState.RUNNING,),
    )
    assert not duplicate
    assert completed.state is AgentJobState.IMPORTING
    assert completed.result == publication()
    retried, duplicate = store.record_result(
        publication(), callback_key="result", expected_revision=job.revision,
        target_state=AgentJobState.IMPORTING, allowed_states=(AgentJobState.RUNNING,),
    )
    assert duplicate
    assert retried.revision == completed.revision


@pytest.fixture
def control(tmp_path):
    control = AgentControlServer.create(tmp_path / "jobs.sqlite3", token="test-token")
    httpd = control.build_http_server()
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield control
    finally:
        httpd.shutdown()
        thread.join(timeout=3)
        httpd.server_close()


def post(control, path, payload):
    request = Request(
        f"http://127.0.0.1:{control.port}/api/v1/{path}",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": "Bearer test-token", "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=3) as response:
        return json.load(response)


@pytest.mark.parametrize("outcome", ["succeeded", "partial", "failed", "cancelled"])
def test_final_publication_http_retry_returns_existing_result(control, outcome):
    running(control.store)
    payload = publication(outcome=outcome).to_dict()
    first = post(control, "jobs/job-1/publications", payload)
    second = post(control, "jobs/job-1/publications", payload)
    assert second["duplicate"] is True
    assert second["revision"] == first["revision"]
    assert second["state"] == first["state"]


def test_conflicting_manifest_returns_http_conflict(control):
    original = manifest()
    post(control, "jobs", original.to_dict())
    with pytest.raises(HTTPError) as error:
        post(control, "jobs", replace(original, actor_id="another-actor").to_dict())
    assert error.value.code == 409


def test_null_expected_revision_returns_structured_conflict(control):
    control.store.enqueue(manifest())
    with pytest.raises(HTTPError) as error:
        post(control, "jobs/job-1/cancel", {"expected_revision": None})
    assert error.value.code == 409
    assert json.load(error.value)["code"] == "agent.invalid_transition"
