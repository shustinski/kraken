from __future__ import annotations

import hashlib

import pytest

from kraken_core.analysis_protocol import (
    AnalysisArtifactInput,
    AnalysisFrameInput,
    AnalysisOutcome,
    AnalysisParameter,
    AnalysisSourceRole,
)
from kraken_core.analysis_run_protocol import (
    ANALYSIS_MAX_EXPRESSION_DEPTH,
    AnalysisExpression,
    AnalysisPartitionJobManifest,
    AnalysisPartitionResultManifest,
    AnalysisRecipe,
    AnalysisRecordBundle,
    AnalysisRunManifest,
    AnalysisRuntimeIdentity,
    AnalysisSourceBinding,
)


def _runtime() -> AnalysisRuntimeIdentity:
    return AnalysisRuntimeIdentity(
        engine_version="1.0.0",
        engine_build="abc123",
        python_version="3.14",
        numpy_version="2.4",
        opencv_version="5.0",
        operating_system="windows",
    )


def _recipe(target: str = "C") -> AnalysisRecipe:
    return AnalysisRecipe(
        AnalysisExpression.binary(
            "compare",
            AnalysisExpression.binary("xor", AnalysisExpression.source("A"), AnalysisExpression.source("B")),
            AnalysisExpression.source(target),
        ),
        metric_keys=("iou", "dice"),
    )


def _frame(frame_id: str = "frame-1") -> AnalysisFrameInput:
    digest = hashlib.sha256(frame_id.encode()).hexdigest()
    return AnalysisFrameInput(
        frame_id=frame_id,
        x=1,
        y=2,
        artifacts=tuple(
            AnalysisArtifactInput(
                binding_key=key,
                role=AnalysisSourceRole.MODEL_OUTPUT,
                artifact_id=f"artifact-{key}",
                artifact_version_id=f"version-{key}-{frame_id}",
                relative_path=f"inputs/{key}/{frame_id}.png",
                media_type="image/png",
                sha256=digest,
            )
            for key in ("A", "B", "C")
        ),
    )


def _run(run_id: str = "run-1") -> AnalysisRunManifest:
    return AnalysisRunManifest(
        run_id=run_id,
        project_id="project-1",
        frame_ids=("frame-1", "frame-2"),
        source_bindings=tuple(
            AnalysisSourceBinding(key, f"model-{key}", "version-1", f"Model {key}") for key in ("A", "B", "C")
        ),
        recipe=_recipe(),
        runtime=_runtime(),
        parameters=(AnalysisParameter("threshold", 0.5),),
    )


def test_expression_contract_supports_required_scenarios() -> None:
    scenarios = (
        AnalysisExpression.binary("compare", AnalysisExpression.source("A"), AnalysisExpression.source("B")),
        _recipe("A").expression,
        _recipe("C").expression,
        AnalysisExpression.binary(
            "compare",
            AnalysisExpression.binary("subtract", AnalysisExpression.source("B"), AnalysisExpression.source("A")),
            AnalysisExpression.source("C"),
        ),
    )

    for expression in scenarios:
        assert AnalysisExpression.from_payload(expression.to_payload()) == expression


def test_expression_rejects_unknown_missing_and_too_deep_operations() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        AnalysisExpression("python")
    with pytest.raises(ValueError, match="requires left and right"):
        AnalysisExpression("xor")

    expression = AnalysisExpression.source("A")
    with pytest.raises(ValueError, match=str(ANALYSIS_MAX_EXPRESSION_DEPTH)):
        for _ in range(ANALYSIS_MAX_EXPRESSION_DEPTH):
            expression = AnalysisExpression.binary("xor", expression, AnalysisExpression.source("B"))


def test_recipe_rejects_nested_compare_and_missing_binding() -> None:
    nested = AnalysisExpression.binary("compare", AnalysisExpression.source("A"), AnalysisExpression.source("B"))
    with pytest.raises(ValueError, match="only allowed"):
        AnalysisRecipe(AnalysisExpression.binary("compare", nested, AnalysisExpression.source("C")))

    with pytest.raises(ValueError, match="missing bindings: C"):
        _recipe().validate_bindings({"A", "B"})


def test_run_round_trip_and_fingerprint_ignore_run_identity(tmp_path) -> None:
    first = _run("run-1")
    second = _run("run-2")
    path = tmp_path / "run.json"
    first.write(path)

    assert AnalysisRunManifest.read(path) == first
    assert first.fingerprint == second.fingerprint
    assert first.recipe.fingerprint == second.recipe.fingerprint


def test_run_partitioning_is_deterministic() -> None:
    frame_ids = tuple(f"frame-{index:06d}" for index in range(100_001))
    run = AnalysisRunManifest(
        run_id="run-1",
        project_id="project-1",
        frame_ids=frame_ids,
        source_bindings=tuple(
            AnalysisSourceBinding(key, f"model-{key}", "v1") for key in ("A", "B", "C")
        ),
        recipe=_recipe(),
        runtime=_runtime(),
    )

    assert run.partition_count == 101
    assert tuple(map(len, run.partition_frame_ids())) == (1000,) * 100 + (1,)
    assert tuple(item for partition in run.partition_frame_ids() for item in partition) == frame_ids


def test_partition_job_validates_every_frame_binding_and_round_trips(tmp_path) -> None:
    run = _run()
    manifest = AnalysisPartitionJobManifest(
        job_id="job-1",
        run_id=run.run_id,
        partition_id="partition-0",
        project_id=run.project_id,
        partition_index=0,
        partition_count=1,
        run_fingerprint=run.fingerprint,
        recipe=run.recipe,
        frames=(_frame(),),
        parameters=run.parameters,
    )
    path = tmp_path / "partition.json"
    manifest.write(path)

    assert AnalysisPartitionJobManifest.read(path) == manifest
    assert manifest.fingerprint

    incomplete = _frame()
    incomplete = AnalysisFrameInput(incomplete.frame_id, incomplete.x, incomplete.y, incomplete.artifacts[:2])
    with pytest.raises(ValueError, match="missing bindings: C"):
        AnalysisPartitionJobManifest(
            job_id="job-2",
            run_id=run.run_id,
            partition_id="partition-0",
            project_id=run.project_id,
            partition_index=0,
            partition_count=1,
            run_fingerprint=run.fingerprint,
            recipe=run.recipe,
            frames=(incomplete,),
        )


def test_partition_result_round_trip(tmp_path) -> None:
    bundle = AnalysisRecordBundle(
        relative_path="outputs/frames.jsonl.gz",
        sha256=hashlib.sha256(b"records").hexdigest(),
        compressed_size=10,
        uncompressed_size=30,
        frame_count=1,
    )
    manifest = AnalysisPartitionResultManifest(
        job_id="job-1",
        run_id="run-1",
        partition_id="partition-0",
        project_id="project-1",
        outcome=AnalysisOutcome.SUCCEEDED,
        bundle=bundle,
    )
    path = tmp_path / "result.json"
    manifest.write(path)

    assert AnalysisPartitionResultManifest.read(path) == manifest

    with pytest.raises(ValueError, match="requires a record bundle"):
        AnalysisPartitionResultManifest(
            job_id="job-1",
            run_id="run-1",
            partition_id="partition-0",
            project_id="project-1",
            outcome=AnalysisOutcome.SUCCEEDED,
            bundle=None,
        )
