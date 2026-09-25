"""Acceptance: 2-layer confidence analytics paints scores and percentiles.

Default size is small for CI. For the real gate:
  set KARAKAL_ACCEPT_FRAMES=20000
  pytest plugins/karakal/tests/performance/test_analytics_accept_2x20k.py -q -s
"""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from karakal.core.analytics import compute_build_result_analytics
from karakal.core.domain import BuildOptions, BuildResult, FrameRecord, GeometryMode, ModelSpec


def _write_gray(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 32), int(value), dtype=np.uint8)
    import cv2

    assert cv2.imwrite(str(path), image)


@pytest.mark.performance
def test_two_layer_confidence_acceptance(tmp_path, monkeypatch) -> None:
    frames = int(os.environ.get("KARAKAL_ACCEPT_FRAMES", "48"))
    workers = int(os.environ.get("KARAKAL_ACCEPT_WORKERS", "4"))
    monkeypatch.setenv("KARAKAL_ANALYTICS_THREAD_POOL", "1")

    root = tmp_path / "synth"
    model_a = "result_direct"
    model_b = "result_inv"
    records: list[FrameRecord] = []
    for index in range(frames):
        key = f"frame_{index:05d}"
        mask_a = root / model_a / f"{key}.png"
        mask_b = root / model_b / f"{key}.png"
        prob_a = root / f"{model_a}_conf" / f"{key}.png"
        prob_b = root / f"{model_b}_conf" / f"{key}.png"
        _write_gray(mask_a, (index * 5) % 255)
        _write_gray(mask_b, (index * 11) % 255)
        _write_gray(prob_a, 20 + (index % 120))
        _write_gray(prob_b, 50 + (index % 100))
        records.append(
            FrameRecord(
                key=key,
                display_name=key,
                model_mask_paths={model_a: str(mask_a), model_b: str(mask_b)},
                model_prob_paths={model_a: str(prob_a), model_b: str(prob_b)},
            )
        )

    specs = (
        ModelSpec(
            model_id=model_a,
            display_name=model_a,
            mask_folder=root / model_a,
            prob_folder=root / f"{model_a}_conf",
        ),
        ModelSpec(
            model_id=model_b,
            display_name=model_b,
            mask_folder=root / model_b,
            prob_folder=root / f"{model_b}_conf",
        ),
    )
    metric_a = f"model_output_confidence::{model_a}"
    metric_b = f"model_output_confidence::{model_b}"
    build = BuildResult(
        records=tuple(records),
        model_specs=specs,
        options=BuildOptions(
            geometry_mode=GeometryMode.MASK,
            analysis_max_side=64,
            max_workers=workers,
            cache_enabled=True,
        ),
        available_metric_keys=(metric_a, metric_b),
        selected_metric_key=metric_a,
    )

    started = perf_counter()
    scored_a = compute_build_result_analytics(build, metric_key=metric_a)
    elapsed_a = perf_counter() - started
    ready_a = sum(1 for record in scored_a.records if record.score_ready)
    pct_a = sum(1 for record in scored_a.records if record.score_percentile is not None)

    started = perf_counter()
    scored_b = compute_build_result_analytics(scored_a, metric_key=metric_b)
    elapsed_b = perf_counter() - started
    ready_b = sum(1 for record in scored_b.records if record.score_ready)
    pct_b = sum(1 for record in scored_b.records if record.score_percentile is not None)

    print(
        f"\naccept_2xN frames={frames} "
        f"layer_a={elapsed_a:.3f}s ({1000 * elapsed_a / max(1, frames):.2f} ms/frame) ready={ready_a} pct={pct_a} "
        f"layer_b={elapsed_b:.3f}s ({1000 * elapsed_b / max(1, frames):.2f} ms/frame) ready={ready_b} pct={pct_b}"
    )
    assert ready_a == frames and pct_a == frames
    assert ready_b == frames and pct_b == frames
