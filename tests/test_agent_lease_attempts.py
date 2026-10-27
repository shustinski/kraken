from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from kraken_agent.jobs import AgentJobState, DurableJobStore, JobStateError
from tests.test_agent_execution_safety import manifest, publication


def test_new_claim_archives_result_progress_callbacks_and_publications_without_reusing_them(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    store = DurableJobStore(path)
    value = manifest()
    job = store.enqueue(value, attempt_key="a" * 64)
    assert store.enqueue(value, attempt_key="a" * 64).revision == job.revision
    job = store.claim_next()
    job = store.transition(job.job_id, AgentJobState.RUNNING, expected_revision=job.revision)
    store.record_progress(job.job_id, {"batch": 123})
    result = publication()
    job, _ = store.record_result(
        result, callback_key="old-result", expected_revision=job.revision, target_state=AgentJobState.IMPORTING
    )
    assert job.result is not None
    current = store.enqueue(value, attempt_key="b" * 64)
    assert current.state is AgentJobState.QUEUED and current.revision > job.revision
    assert current.result is None and current.progress is None and current.error is None
    assert store.list_publications(job.job_id) == ()
    history = DurableJobStore(path).attempt_history(job.job_id)
    assert len(history) == 1
    previous = history[0]
    assert previous["result_json"] == job.result_json and previous["progress_json"] == '{"batch":123}'
    assert previous["attempt_key"] == "a" * 64 and previous["state"] == "importing"
    assert previous["callbacks"][0]["callback_key"] == "old-result"
    assert previous["publications"][0]["publication_id"] == result.publication_id
    with pytest.raises(JobStateError, match="retired"):
        store.enqueue(value, attempt_key="a" * 64)
    assert store.get(job.job_id) == current


def test_new_claim_cannot_replace_an_execution_that_is_still_running(tmp_path):
    store = DurableJobStore(tmp_path / "jobs.sqlite3")
    value = manifest()
    store.enqueue(value, attempt_key="a" * 64)
    active = store.claim_next()
    with pytest.raises(JobStateError, match="still active"):
        store.enqueue(value, attempt_key="b" * 64)
    assert store.get(active.job_id) == active
    assert store.attempt_history(active.job_id) == ()


def test_concurrent_legacy_schema_migration_preserves_existing_job(tmp_path):
    from kraken_agent.protocols import manifest_to_json

    path = tmp_path / "legacy.sqlite3"
    value = manifest()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE jobs(job_id TEXT PRIMARY KEY, state TEXT NOT NULL, manifest_json TEXT NOT NULL, "
            "result_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, revision INTEGER NOT NULL, error TEXT)"
        )
        connection.execute(
            "INSERT INTO jobs VALUES (?, 'cancelled', ?, NULL, '2026-10-25', '2026-10-25', 3, 'interrupted')",
            (value.job_id, manifest_to_json(value)),
        )
    with ThreadPoolExecutor(max_workers=4) as pool:
        stores = tuple(pool.map(lambda _: DurableJobStore(path), range(4)))
    assert all(store.get("job-1").state is AgentJobState.CANCELLED for store in stores)
    renewed = stores[0].enqueue(value, attempt_key="c" * 64)
    assert renewed.state is AgentJobState.QUEUED and renewed.revision == 4
    previous = stores[-1].attempt_history("job-1")
    assert previous[0]["error"] == "interrupted" and previous[0]["revision"] == 3
