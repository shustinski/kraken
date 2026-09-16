from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

from kraken_core.analysis_protocol import (
    AnalysisArtifactInput,
    AnalysisFrameInput,
    AnalysisFrameResult,
    AnalysisMetricValue,
    AnalysisOutcome,
    AnalysisSourceRole,
)
from kraken_core.analysis_run_protocol import (
    AnalysisExpression,
    AnalysisPartitionJobManifest,
    AnalysisPartitionResultManifest,
    AnalysisRecipe,
    AnalysisRecordBundle,
    AnalysisRunManifest,
    AnalysisRuntimeIdentity,
    AnalysisSourceBinding,
    canonical_json,
)
from kraken_manager.infrastructure.analysis import FilesystemAnalysisStore


def _contracts(frame_count: int = 1001):
    frame_ids = tuple(f"frame-{index:04d}" for index in range(frame_count))
    recipe = AnalysisRecipe(
        AnalysisExpression.binary("compare", AnalysisExpression.source("A"), AnalysisExpression.source("B"))
    )
    run = AnalysisRunManifest(
        "run-1",
        "project-1",
        frame_ids,
        (AnalysisSourceBinding("A", "model-a", "v1"), AnalysisSourceBinding("B", "model-b", "v1")),
        recipe,
        AnalysisRuntimeIdentity("1", "build", "3.14", "2", "5", "windows"),
    )
    digest = hashlib.sha256(b"immutable-source").hexdigest()
    partitions = []
    for partition_index, selected in enumerate(run.partition_frame_ids()):
        frames = tuple(
            AnalysisFrameInput(
                frame_id,
                index + 1,
                partition_index + 1,
                tuple(
                    AnalysisArtifactInput(
                        key,
                        AnalysisSourceRole.MODEL_OUTPUT,
                        f"artifact-{key}",
                        f"version-{key}-{frame_id}",
                        f"inputs/{key}/{frame_id}.png",
                        "image/png",
                        digest,
                    )
                    for key in ("A", "B")
                ),
            )
            for index, frame_id in enumerate(selected)
        )
        partitions.append(
            AnalysisPartitionJobManifest(
                f"job-{partition_index}",
                run.run_id,
                f"part-{partition_index}",
                run.project_id,
                partition_index,
                run.partition_count,
                run.fingerprint,
                recipe,
                frames,
            )
        )
    return run, tuple(partitions)


def _result_bundle(root: Path, partition: AnalysisPartitionJobManifest):
    frames = tuple(
        AnalysisFrameResult(
            frame.frame_id,
            frame.x,
            frame.y,
            "ready",
            (
                AnalysisMetricValue("iou", 0.75, 0.75),
                AnalysisMetricValue("xor", 0.25, 0.75, higher_is_better=False),
            ),
        )
        for frame in partition.frames
    )
    path = root / f"{partition.partition_id}.jsonl.gz"
    uncompressed_size = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as archive:
            for frame in frames:
                encoded = (canonical_json(frame.to_payload()) + "\n").encode()
                uncompressed_size += len(encoded)
                archive.write(encoded)
    bundle = AnalysisRecordBundle(
        f"outputs/{path.name}",
        hashlib.sha256(path.read_bytes()).hexdigest(),
        path.stat().st_size,
        uncompressed_size,
        len(frames),
    )
    return path, AnalysisPartitionResultManifest(
        partition.job_id,
        partition.run_id,
        partition.partition_id,
        partition.project_id,
        AnalysisOutcome.SUCCEEDED,
        bundle,
    )


def test_progressive_idempotent_import_and_projection_rebuild(tmp_path) -> None:
    run, partitions = _contracts()
    store = FilesystemAnalysisStore(tmp_path, run.project_id)
    store.create_run(run, partitions)
    second_path, second_result = _result_bundle(tmp_path, partitions[1])
    first_path, first_result = _result_bundle(tmp_path, partitions[0])

    assert store.import_partition(second_result, second_path) is True
    partial = store.get_run(run.run_id)
    assert partial.state == "running"
    assert partial.imported_partitions == 1
    assert len(store.frame_results(run.run_id, "iou")) == 1

    assert store.import_partition(first_result, first_path) is True
    assert store.import_partition(first_result, first_path) is False
    completed = store.get_run(run.run_id)
    assert completed.state == "completed"
    assert completed.completed_frames == 1001
    assert completed.imported_partitions == completed.total_partitions == 2
    assert len(store.frame_results(run.run_id, "iou")) == 1001
    assert store.retryable_partitions(run.run_id) == ()

    store.rebuild()
    rebuilt = store.get_run(run.run_id)
    assert rebuilt == completed
    assert len(store.frame_results(run.run_id, "xor")) == 1001


def test_run_creation_does_not_modify_external_source(tmp_path) -> None:
    external = tmp_path / "external-source.png"
    external.write_bytes(b"immutable-source")
    original_mtime = external.stat().st_mtime_ns
    run, partitions = _contracts(1)

    FilesystemAnalysisStore(tmp_path / "catalog", run.project_id).create_run(run, partitions)

    assert external.read_bytes() == b"immutable-source"
    assert external.stat().st_mtime_ns == original_mtime
