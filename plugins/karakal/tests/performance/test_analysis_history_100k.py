from __future__ import annotations

import hashlib
import os

import pytest

from kraken_core.analysis_protocol import (
    AnalysisArtifactInput,
    AnalysisFrameInput,
    AnalysisFrameResult,
    AnalysisMetricValue,
    AnalysisSourceRole,
)
from kraken_core.analysis_run_protocol import (
    ANALYSIS_PARTITION_SIZE,
    AnalysisExpression,
    AnalysisPartitionJobManifest,
    AnalysisRecipe,
    AnalysisRunManifest,
    AnalysisRuntimeIdentity,
    AnalysisSourceBinding,
)
from karakal.storage import AnalysisHistoryStore


@pytest.mark.performance
@pytest.mark.skipif(os.environ.get("KARAKAL_RUN_100K_TEST") != "1", reason="explicit 100k storage gate")
def test_standalone_history_stores_100k_frames(tmp_path) -> None:
    total_frames = 100_000
    frame_ids = tuple(f"frame-{index:06d}" for index in range(total_frames))
    recipe = AnalysisRecipe(
        AnalysisExpression.binary("compare", AnalysisExpression.source("A"), AnalysisExpression.source("B")),
        metric_keys=("iou",),
    )
    run = AnalysisRunManifest(
        "run-100k",
        "standalone",
        frame_ids,
        (AnalysisSourceBinding("A", "model-a", "v1"), AnalysisSourceBinding("B", "model-b", "v1")),
        recipe,
        AnalysisRuntimeIdentity("1", "build", "3.14", "2", "5", "performance"),
    )
    digest = hashlib.sha256(b"synthetic").hexdigest()
    store = AnalysisHistoryStore(tmp_path / "history.sqlite3")
    store.create_run(run)

    for partition_index, selected in enumerate(run.partition_frame_ids()):
        frames = tuple(
            AnalysisFrameInput(
                frame_id,
                offset + 1,
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
            for offset, frame_id in enumerate(selected)
        )
        partition = AnalysisPartitionJobManifest(
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
        store.save_partition(partition)
        store.import_partition(
            partition.partition_id,
            tuple(
                AnalysisFrameResult(
                    frame.frame_id,
                    frame.x,
                    frame.y,
                    "ready",
                    (AnalysisMetricValue("iou", 1.0, 1.0),),
                )
                for frame in frames
            ),
        )

    completed = store.get_run(run.run_id)
    assert completed.state == "completed"
    assert completed.completed_frames == total_frames
    assert len(store.frame_results(run.run_id, metric_key="iou")) == total_frames
    assert run.partition_count == total_frames // ANALYSIS_PARTITION_SIZE
