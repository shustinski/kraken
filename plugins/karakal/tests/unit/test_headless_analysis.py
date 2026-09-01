from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from kraken_core.analysis_protocol import AnalysisArtifactInput, AnalysisFrameInput, AnalysisSourceRole
from kraken_core.analysis_run_protocol import AnalysisExpression, AnalysisPartitionJobManifest, AnalysisRecipe
from karakal.core.headless import render_analysis_map, run_analysis


def _write_mask(path: Path, values: list[list[int]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.asarray(values, dtype=np.uint8) * 255)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _job(tmp_path: Path, expression: AnalysisExpression) -> AnalysisPartitionJobManifest:
    artifacts = []
    values = {
        "A": [[1, 1], [0, 0]],
        "B": [[1, 0], [1, 0]],
        "C": [[0, 1], [1, 0]],
    }
    for key, mask in values.items():
        relative = f"inputs/{key}.png"
        digest = _write_mask(tmp_path / relative, mask)
        artifacts.append(
            AnalysisArtifactInput(
                binding_key=key,
                role=AnalysisSourceRole.MODEL_OUTPUT,
                artifact_id=f"artifact-{key}",
                artifact_version_id=f"version-{key}",
                relative_path=relative,
                media_type="image/png",
                sha256=digest,
            )
        )
    return AnalysisPartitionJobManifest(
        job_id="job-1",
        run_id="run-1",
        partition_id="part-0",
        project_id="project-1",
        partition_index=0,
        partition_count=1,
        run_fingerprint="fingerprint",
        recipe=AnalysisRecipe(expression),
        frames=(AnalysisFrameInput("frame-1", 1, 1, tuple(artifacts)),),
    )


def _compare(left: AnalysisExpression, right: AnalysisExpression) -> AnalysisExpression:
    return AnalysisExpression.binary("compare", left, right)


def test_headless_direct_xor_and_directional_subtraction(tmp_path) -> None:
    direct = run_analysis(
        _job(tmp_path, _compare(AnalysisExpression.source("A"), AnalysisExpression.source("B"))),
        tmp_path,
        tmp_path / "out",
    )
    xor_target = run_analysis(
        _job(
            tmp_path,
            _compare(
                AnalysisExpression.binary("xor", AnalysisExpression.source("A"), AnalysisExpression.source("B")),
                AnalysisExpression.source("C"),
            ),
        ),
        tmp_path,
        tmp_path / "out",
    )
    b_minus_a = run_analysis(
        _job(
            tmp_path,
            _compare(
                AnalysisExpression.binary("subtract", AnalysisExpression.source("B"), AnalysisExpression.source("A")),
                AnalysisExpression.source("A"),
            ),
        ),
        tmp_path,
        tmp_path / "out",
    )
    a_minus_b = run_analysis(
        _job(
            tmp_path,
            _compare(
                AnalysisExpression.binary("subtract", AnalysisExpression.source("A"), AnalysisExpression.source("B")),
                AnalysisExpression.source("A"),
            ),
        ),
        tmp_path,
        tmp_path / "out",
    )

    assert {metric.key: metric.raw_value for metric in direct.frames[0].metrics} == {
        "xor": 0.5,
        "iou": 1 / 3,
        "dice": 0.5,
    }
    assert {metric.key: metric.raw_value for metric in xor_target.frames[0].metrics}["iou"] == 1.0
    assert b_minus_a.frames[0].metrics != a_minus_b.frames[0].metrics


def test_headless_progress_cancellation_and_decode_failure(tmp_path) -> None:
    job = _job(tmp_path, _compare(AnalysisExpression.source("A"), AnalysisExpression.source("B")))
    progress = []
    result = run_analysis(job, tmp_path, tmp_path / "out", lambda *items: progress.append(items))
    cancelled = run_analysis(job, tmp_path, tmp_path / "out", cancellation=lambda: True)
    (tmp_path / "inputs" / "A.png").write_bytes(b"broken")
    broken = run_analysis(job, tmp_path, tmp_path / "out")

    assert result.outcome.value == "succeeded"
    assert progress == [(1, 1, "frame-1")]
    assert cancelled.outcome.value == "cancelled"
    assert broken.outcome.value == "failed"
    assert broken.frames[0].status == "not_computed"
    assert broken.frames[0].message


def test_render_map_is_cached(tmp_path) -> None:
    job = _job(tmp_path, _compare(AnalysisExpression.source("A"), AnalysisExpression.source("B")))
    first = render_analysis_map(job, "frame-1", tmp_path, tmp_path / "cache")
    modified = first.stat().st_mtime_ns
    second = render_analysis_map(job, "frame-1", tmp_path, tmp_path / "cache")

    assert first == second
    assert first.stat().st_mtime_ns == modified
    assert cv2.imread(str(first)) is not None
