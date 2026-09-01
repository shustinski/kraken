from __future__ import annotations

import hashlib

from kraken_core.analysis_protocol import (
    AnalysisArtifactInput,
    AnalysisFrameInput,
    AnalysisFrameResult,
    AnalysisMetricValue,
    AnalysisSourceRole,
)
from kraken_core.analysis_run_protocol import (
    AnalysisExpression,
    AnalysisPartitionJobManifest,
    AnalysisRecipe,
    AnalysisRunManifest,
    AnalysisRuntimeIdentity,
    AnalysisSourceBinding,
)
from karakal.storage.analysis_history import AnalysisHistoryStore, SCHEMA_VERSION, default_history_database


def _run(run_id: str) -> AnalysisRunManifest:
    return AnalysisRunManifest(
        run_id=run_id,
        project_id="standalone",
        frame_ids=("frame-1",),
        source_bindings=(
            AnalysisSourceBinding("A", "model-a", "1"),
            AnalysisSourceBinding("B", "model-b", "1"),
        ),
        recipe=AnalysisRecipe(
            AnalysisExpression.binary("compare", AnalysisExpression.source("A"), AnalysisExpression.source("B"))
        ),
        runtime=AnalysisRuntimeIdentity("1", "build", "3.14", "2", "5", "windows"),
    )


def _partition(run: AnalysisRunManifest) -> AnalysisPartitionJobManifest:
    digest = hashlib.sha256(b"mask").hexdigest()
    frame = AnalysisFrameInput(
        "frame-1",
        1,
        2,
        tuple(
            AnalysisArtifactInput(
                key,
                AnalysisSourceRole.MODEL_OUTPUT,
                f"artifact-{key}",
                f"version-{key}",
                f"inputs/{key}.png",
                "image/png",
                digest,
            )
            for key in ("A", "B")
        ),
    )
    return AnalysisPartitionJobManifest(
        "job-1",
        run.run_id,
        "part-0",
        run.project_id,
        0,
        1,
        run.fingerprint,
        run.recipe,
        (frame,),
    )


def test_default_database_honours_data_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("KARAKAL_DATA_DIR", str(tmp_path))
    assert default_history_database() == tmp_path / "analysis.sqlite3"


def test_migration_is_repeatable_and_runs_keep_reproducibility_fingerprint(tmp_path) -> None:
    store = AnalysisHistoryStore(tmp_path / "history.sqlite3")
    store.migrate()
    first = _run("run-1")
    second = _run("run-2")
    store.create_run(first)
    store.create_run(second)

    assert store.schema_version() == SCHEMA_VERSION
    assert len(store.list_runs()) == 2
    assert store.get_run("run-1").fingerprint == store.get_run("run-2").fingerprint


def test_partition_import_is_idempotent_and_recovery_lists_only_incomplete(tmp_path) -> None:
    store = AnalysisHistoryStore(tmp_path / "history.sqlite3")
    run = _run("run-1")
    partition = _partition(run)
    store.create_run(run)
    store.save_partition(partition)
    assert [item.partition_id for item in store.incomplete_partitions(run.run_id)] == ["part-0"]
    store.mark_partition_running(partition.partition_id)
    frame = AnalysisFrameResult(
        "frame-1",
        1,
        2,
        "ready",
        (AnalysisMetricValue("iou", 0.75, 0.75), AnalysisMetricValue("xor", 0.25, 0.75, higher_is_better=False)),
    )

    assert store.import_partition(partition.partition_id, (frame,)) is True
    assert store.import_partition(partition.partition_id, (frame,)) is False
    assert store.incomplete_partitions(run.run_id) == ()
    assert store.get_run(run.run_id).state == "completed"
    assert len(store.frame_results(run.run_id)) == 2


def test_delete_run_cascades_history(tmp_path) -> None:
    store = AnalysisHistoryStore(tmp_path / "history.sqlite3")
    run = _run("run-1")
    store.create_run(run)
    store.save_partition(_partition(run))

    assert store.delete_run(run.run_id) is True
    assert store.get_run(run.run_id) is None
    assert store.frame_results(run.run_id) == ()
