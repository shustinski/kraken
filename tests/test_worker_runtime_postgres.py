"""Runtime import guards on real PostgreSQL; generated record fixtures test storage only."""
from dataclasses import replace

import pytest

from karakal.runtime_identity import worker_runtime_identity
from kraken_core.analysis_bundle import stream_bundle_records
from kraken_manager.infrastructure.postgres.analysis_store import PostgresAnalysisProjectionStore
from tests.test_admin_project_workflows import project
from tests.test_analysis_filesystem_store import _contracts, _result_bundle

pytest_plugins = ["tests.test_admin_project_workflows"]


def test_pg_import_pins_worker_and_preserves_original_rows_on_mismatch(database, tmp_path):
    services, _accounts, owner, _config = database
    project_id = project(services, owner)["project_id"]
    original, parts = _contracts(1001)
    run = replace(original, project_id=project_id)
    parts = tuple(replace(part, project_id=project_id, run_fingerprint=run.fingerprint, require_worker_runtime=True) for part in parts)
    store = PostgresAnalysisProjectionStore(services.engine)
    store.create_run(run, parts)
    runtime = worker_runtime_identity()
    results = []
    for part in parts:
        path, result = _result_bundle(tmp_path, part)
        with path.open("rb") as stream:
            frames = tuple(stream_bundle_records(stream, result.bundle, expected_frame_ids=(f.frame_id for f in part.frames)))
        results.append((replace(result, worker_runtime=runtime), frames))
    first, frames = results[0]
    assert store.import_partition(first, frames)
    other, records = results[1]
    incompatible = replace(other, worker_runtime=replace(runtime, opencv_version="incompatible-fixture"))
    with pytest.raises(ValueError, match="incompatible worker runtimes"):
        store.import_partition(incompatible, records)
    assert len(store.frame_results(run.run_id, "iou")) == 1000
    assert store.get_run(run.run_id).worker_runtime == runtime
    with pytest.raises(ValueError, match="receipt is missing"):
        store.import_partition(replace(other, worker_runtime=None), records)
    assert store.import_partition(other, records)
    assert store.get_run(run.run_id).state == "completed"
    assert len(store.frame_results(run.run_id, "iou")) == 1001
    import sqlalchemy as sa
    with services.engine.begin() as connection:
        connection.execute(sa.update(store.partitions).where(store.partitions.c.partition_id == parts[0].partition_id).values(result=None))
    assert store.get_run(run.run_id).worker_runtime is None
    with services.engine.begin() as connection:
        connection.execute(sa.update(store.partitions).where(store.partitions.c.partition_id == parts[0].partition_id).values(result=first.to_payload()))
    with pytest.raises(ValueError):
        store.import_partition(incompatible, records)
    assert store.get_run(run.run_id).worker_runtime == runtime
