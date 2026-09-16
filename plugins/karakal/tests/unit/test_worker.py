from __future__ import annotations

import gzip
import hashlib
import json

import cv2
import numpy as np

from kraken_core.analysis_bundle import stream_bundle_records
from kraken_core.analysis_protocol import AnalysisArtifactInput, AnalysisFrameInput, AnalysisSourceRole
from kraken_core.analysis_run_protocol import (
    AnalysisExpression,
    AnalysisPartitionJobManifest,
    AnalysisPartitionResultManifest,
    AnalysisRecipe,
)
from karakal.worker import execute, main


def _job(tmp_path) -> AnalysisPartitionJobManifest:
    artifacts = []
    for key, values in {"A": [[1, 0]], "B": [[0, 1]]}.items():
        relative = f"inputs/{key}.png"
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        assert cv2.imwrite(str(path), np.asarray(values, dtype=np.uint8) * 255)
        artifacts.append(
            AnalysisArtifactInput(
                key,
                AnalysisSourceRole.MODEL_OUTPUT,
                f"artifact-{key}",
                f"version-{key}",
                relative,
                "image/png",
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return AnalysisPartitionJobManifest(
        "job-1",
        "run-1",
        "part-0",
        "project-1",
        0,
        1,
        "fingerprint",
        AnalysisRecipe(
            AnalysisExpression.binary("compare", AnalysisExpression.source("A"), AnalysisExpression.source("B"))
        ),
        (AnalysisFrameInput("frame-1", 1, 1, tuple(artifacts)),),
    )


def test_worker_writes_atomic_progress_bundle_and_result(tmp_path) -> None:
    job = _job(tmp_path)
    job_path = tmp_path / "job.json"
    result_path = tmp_path / "result.json"
    progress_path = tmp_path / "progress.json"
    job.write(job_path)

    result = execute(job_path, result_path, progress_path, tmp_path)

    assert AnalysisPartitionResultManifest.read(result_path) == result
    assert json.loads(progress_path.read_text())["completed_frames"] == 1
    bundle_path = tmp_path / result.bundle.relative_path
    assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == result.bundle.sha256
    with bundle_path.open("rb") as stream:
        frames = tuple(stream_bundle_records(stream, result.bundle, expected_frame_ids=("frame-1",)))
    assert frames[0].status == "ready"
    assert {metric.key for metric in frames[0].metrics} == {"xor", "iou", "dice"}


def test_worker_cancel_and_invalid_manifest_exit_codes(tmp_path, monkeypatch) -> None:
    job = _job(tmp_path)
    job_path = tmp_path / "job.json"
    job.write(job_path)
    cancel_path = tmp_path / "cancel"
    cancel_path.touch()
    result = execute(job_path, tmp_path / "cancelled.json", tmp_path / "progress.json", tmp_path, cancel_path)
    assert result.outcome.value == "cancelled"

    broken = tmp_path / "broken.json"
    broken.write_text("{}")
    monkeypatch.setenv("KRAKEN_JOB_MANIFEST", str(broken))
    monkeypatch.setenv("KRAKEN_RESULT_MANIFEST", str(tmp_path / "result.json"))
    monkeypatch.setenv("KRAKEN_PROGRESS_PATH", str(tmp_path / "progress.json"))
    monkeypatch.setenv("KRAKEN_STAGING_ROOT", str(tmp_path))
    assert main([]) == 2


def test_bundle_validator_rejects_tampered_count(tmp_path) -> None:
    job = _job(tmp_path)
    job_path = tmp_path / "job.json"
    job.write(job_path)
    result = execute(job_path, tmp_path / "result.json", tmp_path / "progress.json", tmp_path)
    bundle_path = tmp_path / result.bundle.relative_path
    with gzip.open(bundle_path, "rb") as stream:
        assert stream.readline()
    tampered = type(result.bundle)(
        result.bundle.relative_path,
        result.bundle.sha256,
        result.bundle.compressed_size,
        result.bundle.uncompressed_size,
        2,
    )
    with bundle_path.open("rb") as stream:
        try:
            tuple(stream_bundle_records(stream, tampered, expected_frame_ids=("frame-1",)))
        except ValueError as exc:
            assert "frame count" in str(exc)
        else:
            raise AssertionError("tampered bundle was accepted")
