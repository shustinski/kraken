"""Restart recovery using durable stores, real subprocesses and portable files."""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import textwrap
import threading
import time
from uuid import uuid4

import pytest
import sqlalchemy as sa

from kraken_agent.jobs import AgentJobState, DurableJobStore, StagingWorkspace
from kraken_agent.runner import PluginRegistry, SubprocessPluginRunner
from kraken_agent.server_worker import ServerAgentWorker
from kraken_core.job_recovery import execution_state, progress_fraction
from kraken_server.agent_auth import AgentIdentity
from kraken_server.agent_gateway import PostgresAgentGateway
from tests.test_agent_execution_safety import manifest, staged_runner


def test_local_restart_passes_durable_checkpoint_to_actual_plugin(tmp_path):
    script = textwrap.dedent("""
        import hashlib, json, os, time
        from pathlib import Path
        from kraken_core.plugin_protocol import PluginResultManifest
        root = Path(os.environ['KRAKEN_STAGING_ROOT'])
        resume = json.loads(Path(os.environ['KRAKEN_RESUME_PATH']).read_text())
        if resume['checkpoint'] is None:
            path = root / 'checkpoints/model.bin'
            path.parent.mkdir()
            path.write_bytes(b'saved computation')
            checkpoint = {'job_id': 'job-1', 'state': {'next_batch': 17}, 'files': [{
                'relative_path': 'checkpoints/model.bin', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'size_bytes': path.stat().st_size}]}
            Path(os.environ['KRAKEN_CHECKPOINT_PATH']).write_text(json.dumps(checkpoint))
            Path(os.environ['KRAKEN_PROGRESS_PATH']).write_text(json.dumps({'job_id':'job-1','fraction':0.4}))
            while True:
                time.sleep(0.1)
        assert resume['checkpoint']['state']['next_batch'] == 17
        assert resume['progress']['fraction'] == 0.4
        assert (root / 'checkpoints/model.bin').read_bytes() == b'saved computation'
        (root / 'resumed').touch()
        result = PluginResultManifest(job_id='job-1', outcome='partial', plugin_id='test',
                                      plugin_version='1', outputs=())
        Path(os.environ['KRAKEN_RESULT_MANIFEST']).write_text(result.to_json())
    """)
    store, runner, workspace = staged_runner(tmp_path, script)
    execution = threading.Thread(target=runner.run_once)
    execution.start()
    try:
        deadline = time.monotonic() + 10
        while store.get("job-1").checkpoint is None or store.get("job-1").progress is None:
            assert execution.is_alive(), store.get("job-1").error
            assert time.monotonic() < deadline
            time.sleep(0.02)
    finally:
        runner.stop()
        execution.join(5)
    assert not execution.is_alive()
    reopened = DurableJobStore(store.database)
    assert reopened.resume_interrupted() == 1
    assert reopened.get("job-1").progress["fraction"] == 0.4
    new_runner = SubprocessPluginRunner(reopened, runner.staging_root, runner.registry)
    assert new_runner.run_once()
    assert (workspace.path / "resumed").exists(), reopened.get("job-1").error
    assert reopened.get("job-1").state is AgentJobState.PARTIAL


def test_ready_result_is_not_reexecuted_on_local_restart(tmp_path):
    from kraken_core.plugin_protocol import PluginResultManifest

    store = DurableJobStore(tmp_path / "jobs.sqlite3")
    job = store.enqueue(manifest())
    job = store.claim_next()
    job = store.transition(job.job_id, AgentJobState.RUNNING, expected_revision=job.revision)
    result = PluginResultManifest(job_id=job.job_id, outcome="succeeded", plugin_id="test",
                                  plugin_version="1", outputs=())
    store.record_result(result, callback_key="finished", expected_revision=job.revision,
                        target_state=AgentJobState.IMPORTING)
    reopened = DurableJobStore(store.database)
    assert reopened.resume_interrupted() == 0
    assert reopened.get(job.job_id).state is AgentJobState.IMPORTING
    assert reopened.get(job.job_id).result == result
    assert reopened.claim_next() is None


@pytest.fixture
def persisted_queue(tmp_path):
    # Exercise SQL persistence and claim checks without requiring a running PG.
    # Concurrent PostgreSQL locking has separate integration coverage.
    engine = sa.create_engine("sqlite:///" + str(tmp_path / "server.sqlite3"))
    queue = PostgresAgentGateway(engine, None, None)
    queue.metadata.create_all(engine)
    job_id, project_id = str(uuid4()), str(uuid4())
    now = datetime.now(UTC)
    with engine.begin() as connection:
        connection.execute(sa.insert(queue.jobs).values(job_id=job_id, project_id=project_id, state="leased",
            payload={"capability": "frames.vectorize.v1", "parameters": {}}, lease_owner="agent",
            lease_claim="first", lease_until=now + timedelta(minutes=1), lease_attempts=1,
            created_at=now, updated_at=now))
    identity = AgentIdentity("agent", "test", frozenset({"frames.vectorize.v1"}), lease_claim="first")
    yield queue, identity, job_id, project_id
    engine.dispose()


def test_server_state_survives_reclaim_and_rejects_old_agent(persisted_queue):
    queue, identity, job_id, project_id = persisted_queue
    queue.save_execution_state(job_id, identity, {"progress": {"fraction": 0.6}})
    with queue.engine.begin() as connection:
        connection.execute(sa.update(queue.jobs).where(queue.jobs.c.job_id == job_id).values(
            lease_until=datetime.now(UTC) - timedelta(seconds=1)))
    reopened = PostgresAgentGateway(queue.engine, None, None)
    lease = reopened.lease(replace(identity, lease_claim=None))
    assert lease["execution_state"] == {"progress": {"fraction": 0.6}}
    with pytest.raises(PermissionError):
        queue.save_execution_state(job_id, identity, {"progress": {"fraction": 0.1}})
    assert reopened.project_execution_states(project_id)[job_id]["progress"]["fraction"] == 0.6


def test_server_publication_is_imported_after_restart_without_rerunning_plugin(persisted_queue, tmp_path, monkeypatch):
    queue, identity, job_id, _ = persisted_queue
    with queue.engine.begin() as connection:
        connection.execute(sa.insert(queue.publications).values(job_id=job_id, publication_id="ready",
            payload={"outcome": "succeeded"}, created_at=datetime.now(UTC)))
        connection.execute(sa.update(queue.jobs).where(queue.jobs.c.job_id == job_id).values(
            lease_until=datetime.now(UTC) - timedelta(seconds=1)))
    lease = queue.lease(replace(identity, lease_claim=None))
    assert lease["result_ready"] is True
    worker = ServerAgentWorker(server_url="http://127.0.0.1", token="test", data_dir=tmp_path / "new-agent",
                               registry=PluginRegistry())
    monkeypatch.setattr(worker, "_heartbeat", lambda *args: None)
    calls = []

    def request(method, path, payload):
        calls.append(path)
        assert path.endswith("/complete")
        return {"state": "completed"}

    monkeypatch.setattr(worker, "_request", request)
    monkeypatch.setattr(worker.runner, "run_once", lambda **kwargs: pytest.fail("result is already on server"))
    assert worker.run_leased_job(lease) == {"state": "completed"}
    assert len(calls) == 1


def test_running_job_progress_updates_without_a_state_change():
    from kraken_manager.application.dto import CommandContext, SynchronizePluginJobCommand
    from kraken_manager.application.plugin_jobs import SynchronizePluginJobHandler
    from tests.test_plugin_job_application import Fixture, Clock

    fixture = Fixture()
    job = fixture.submit()
    handler = SynchronizePluginJobHandler(fixture.factory, fixture.profiles, Clock())

    def update(current, fraction):
        return handler(SynchronizePluginJobCommand(
            context=CommandContext(actor=fixture.actor, idempotency_key=str(uuid4())),
            project_id=current.project_id, job_id=current.id, expected_revision=current.revision,
            state="running", progress=fraction))

    running = update(job, 0.2)
    progressed = update(running, 0.6)
    assert progressed.state is running.state
    assert progressed.progress == 0.6
    assert progressed.revision > running.revision
    assert update(progressed, 0.6) == progressed


def test_checkpoint_files_restore_on_a_different_machine(persisted_queue, tmp_path, monkeypatch):
    queue, identity, job_id, _ = persisted_queue
    content = b"model and optimizer state"
    digest = hashlib.sha256(content).hexdigest()
    checkpoint = {"job_id": job_id, "state": {"next_batch": 23}, "files": [{
        "relative_path": "checkpoints/model.bin", "sha256": digest, "size_bytes": len(content)}]}
    with pytest.raises(ValueError, match="not been uploaded"):
        queue.save_execution_state(job_id, identity, {"checkpoint": checkpoint})
    with queue.engine.begin() as connection:
        connection.execute(sa.insert(queue.outputs).values(job_id=job_id, output_id="checkpoint-" + digest,
            sha256=digest, size_bytes=len(content), created_at=datetime.now(UTC)))
    queue.save_execution_state(job_id, identity, {"progress": {"fraction": 0.7}, "checkpoint": checkpoint})
    # A fresh data directory has no previous local jobs or staging files.
    worker = ServerAgentWorker(server_url="http://127.0.0.1", token="test", data_dir=tmp_path / "fresh-machine",
                               registry=PluginRegistry())
    worker.store.enqueue(replace(manifest(), job_id=job_id))
    workspace = StagingWorkspace(worker.runner.staging_root, job_id)
    workspace.create()

    def download(url, path, **kwargs):
        reference = queue.checkpoint_blob(job_id, url.rsplit("/", 1)[-1], identity)
        assert reference.sha256 == digest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    monkeypatch.setattr(worker, "_download", download)
    with queue.engine.connect() as connection:
        state = connection.execute(sa.select(queue.jobs.c.execution_state)).scalar_one()
    worker._restore_execution_state(job_id, workspace, state)
    assert workspace.resolve("checkpoints/model.bin").read_bytes() == content
    assert worker.store.get(job_id).checkpoint == checkpoint
    assert worker.store.get(job_id).progress == {"fraction": 0.7}
    with pytest.raises(PermissionError):
        queue.checkpoint_blob(job_id, "f" * 64, identity)


@pytest.mark.parametrize("path", ["../escape", "inputs/overwrite.png", "checkpoints/../escape"])
def test_checkpoint_paths_cannot_overwrite_inputs_or_escape(path):
    with pytest.raises(ValueError):
        execution_state({"checkpoint": {"files": [{"relative_path": path, "sha256": "a" * 64, "size_bytes": 1}]}})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1, 2])
def test_invalid_progress_is_not_published_as_a_fraction(value):
    assert progress_fraction({"fraction": value}, 0.3) == 0.3
