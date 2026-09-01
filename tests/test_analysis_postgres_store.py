from __future__ import annotations

import hashlib

import pytest

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
)


def test_postgres_analysis_projection_contract_matches_sqlite_semantics() -> None:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from kraken_manager.infrastructure.postgres.analysis_store import PostgresAnalysisProjectionStore

    project_id = "00000000-0000-0000-0000-000000000001"
    recipe = AnalysisRecipe(
        AnalysisExpression.binary("compare", AnalysisExpression.source("A"), AnalysisExpression.source("B"))
    )
    run = AnalysisRunManifest(
        "run-1",
        project_id,
        ("frame-1",),
        (AnalysisSourceBinding("A", "model-a", "v1"), AnalysisSourceBinding("B", "model-b", "v1")),
        recipe,
        AnalysisRuntimeIdentity("1", "build", "3.14", "2", "5", "test"),
    )
    digest = hashlib.sha256(b"mask").hexdigest()
    frame = AnalysisFrameInput(
        "frame-1",
        1,
        1,
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
    partition = AnalysisPartitionJobManifest(
        "job-1", run.run_id, "part-0", project_id, 0, 1, run.fingerprint, recipe, (frame,)
    )
    bundle = AnalysisRecordBundle("outputs/frames.jsonl.gz", digest, 1, 1, 1)
    result = AnalysisPartitionResultManifest(
        partition.job_id,
        run.run_id,
        partition.partition_id,
        project_id,
        AnalysisOutcome.SUCCEEDED,
        bundle,
    )
    frame_result = AnalysisFrameResult(
        "frame-1", 1, 1, "ready", (AnalysisMetricValue("iou", 1.0, 1.0),)
    )
    store = PostgresAnalysisProjectionStore(
        sqlalchemy.create_engine("sqlite+pysqlite:///:memory:"), create_schema_for_tests=True
    )

    store.create_run(run, (partition,))
    assert store.import_partition(result, (frame_result,)) is True
    assert store.import_partition(result, (frame_result,)) is False
    payload = store.get_run_payload(run.run_id)
    assert payload["state"] == "completed"
    assert payload["completed_frames"] == 1
