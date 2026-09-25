"""Large-matrix prepare path and worker busy-state recovery."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pytest

from karakal.core.analytics import collect_frame_records, compute_build_result_analytics
from karakal.core.diagnostics import install_diagnostics, pack_diagnostics_zip, stage_enter, stage_exit
from karakal.core.domain import BuildOptions, ModelSpec
from karakal.core.repository_shared import BuildCancelledError
from karakal.core.workers import AnalyticsWorker, FrameIndexWorker


def _write_layers(root: Path, frames: int) -> tuple[Path, Path, Path, Path]:
    folders = [root / "m1", root / "m2", root / "c1", root / "c2"]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    blob = cv2.imencode(".png", np.zeros((8, 8), dtype=np.uint8))[1].tobytes()
    for index in range(frames):
        name = f"frame_{index:05d}.png"
        for folder in folders:
            (folder / name).write_bytes(blob)
    return folders[0], folders[1], folders[2], folders[3]


def test_prepare_20000_frames_with_confidence_stays_fast(tmp_path: Path) -> None:
    m1, m2, c1, c2 = _write_layers(tmp_path, 20_000)
    specs = (
        ModelSpec(model_id="m1", display_name="M1", mask_folder=m1, prob_folder=c1),
        ModelSpec(model_id="m2", display_name="M2", mask_folder=m2, prob_folder=c2),
    )
    started = perf_counter()
    result = collect_frame_records(specs, BuildOptions())
    elapsed = perf_counter() - started
    assert len(result.records) == 20_000
    # Local SSD budget: previously ~40s due to per-key Path.resolve(); target is seconds.
    assert elapsed < 25.0, f"prepare took {elapsed:.2f}s"


def test_scenario_b_compute_starts_after_index_only(tmp_path: Path) -> None:
    m1, m2, c1, c2 = _write_layers(tmp_path, 64)
    specs = (
        ModelSpec(model_id="m1", display_name="M1", mask_folder=m1, prob_folder=c1),
        ModelSpec(model_id="m2", display_name="M2", mask_folder=m2, prob_folder=c2),
    )
    indexed = collect_frame_records(specs, BuildOptions())
    assert indexed.scores_computed is False
    progress: list[tuple[int, int]] = []

    def on_progress(current: int, total: int, _key: str) -> None:
        progress.append((current, total))

    computed = compute_build_result_analytics(
        indexed,
        metric_key="overall_frame_score",
        progress_callback=on_progress,
    )
    assert computed.scores_computed is True
    assert progress
    assert progress[-1][0] == progress[-1][1] == 64


def test_analytics_worker_failure_emits_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m1, m2, c1, c2 = _write_layers(tmp_path, 8)
    specs = (
        ModelSpec(model_id="m1", display_name="M1", mask_folder=m1, prob_folder=c1),
        ModelSpec(model_id="m2", display_name="M2", mask_folder=m2, prob_folder=c2),
    )
    indexed = collect_frame_records(specs, BuildOptions())
    worker = AnalyticsWorker(indexed, "overall_frame_score")
    failures: list[str] = []
    worker.failed.connect(failures.append)

    def boom(*_args, **_kwargs):
        raise RuntimeError("synthetic worker crash")

    monkeypatch.setattr("karakal.core.workers.compute_build_result_analytics", boom)
    worker.run()
    assert failures and "synthetic worker crash" in failures[0]


def test_index_worker_cancel_emits_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m1, m2, c1, c2 = _write_layers(tmp_path, 8)
    specs = (
        ModelSpec(model_id="m1", display_name="M1", mask_folder=m1, prob_folder=c1),
        ModelSpec(model_id="m2", display_name="M2", mask_folder=m2, prob_folder=c2),
    )
    worker = FrameIndexWorker(specs, BuildOptions(), None)
    cancelled: list[bool] = []
    worker.cancelled.connect(lambda: cancelled.append(True))

    def cancel_immediately(*_args, **_kwargs):
        raise BuildCancelledError("Build cancelled")

    monkeypatch.setattr("karakal.core.workers.collect_frame_records", cancel_immediately)
    worker.run()
    assert cancelled == [True]


def test_diagnostics_pack_and_hang_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = install_diagnostics(hang_interval_seconds=30.0)
    stage_enter("synthetic.sleep", frames=1, pid=os.getpid())
    stage_exit("synthetic.sleep", status="ok")
    assert (log_dir / "app.log").is_file()
    packed = pack_diagnostics_zip(tmp_path / "diag.zip", version="test", launch_args=["karakal"])
    assert packed.is_file() and packed.stat().st_size > 0
