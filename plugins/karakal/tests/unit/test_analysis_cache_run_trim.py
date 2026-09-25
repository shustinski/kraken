from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from karakal.core import analysis_cache as cache_mod
from karakal.core.analysis_cache import (
    begin_analysis_cache_run,
    end_analysis_cache_run,
    _store_cached_record_payload,
    _trim_pickle_cache_dir,
)
from karakal.core.analytics import compute_build_result_analytics
from karakal.core.domain import (
    BuildOptions,
    BuildResult,
    FrameRecord,
    GeometryMode,
    ModelSpec,
)


@pytest.fixture()
def isolated_analysis_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "analysis_cache"
    cache_dir.mkdir()
    monkeypatch.setattr(cache_mod, "ANALYSIS_CACHE_DIR", cache_dir)
    monkeypatch.setattr(cache_mod, "_PICKLE_CACHE_BYTES", {})
    monkeypatch.setattr(cache_mod, "_PICKLE_CACHE_FILES", {})
    monkeypatch.setattr(cache_mod, "_PICKLE_CACHE_WRITES_SINCE_TRIM", {})
    monkeypatch.setattr(cache_mod, "_PICKLE_CACHE_PROTECTED_KEYS", set())
    monkeypatch.setattr(cache_mod, "_PICKLE_CACHE_RUN_DEPTH", 0)
    cache_mod._CACHE_TRIM_LAST_BY_DIR.clear()
    return cache_dir


def test_protected_run_keys_not_trimmed_during_run(isolated_analysis_cache, monkeypatch) -> None:
    cache_dir = isolated_analysis_cache

    class _Perf:
        disk_cache_limit_mb = 0

    monkeypatch.setattr(cache_mod, "_active_performance_config", lambda: _Perf())
    monkeypatch.setattr(cache_mod, "_CACHE_TRIM_EVERY_N_WRITES", 1)

    begin_analysis_cache_run()
    run_keys = [f"run-{index}" for index in range(6)]
    for key in run_keys:
        _store_cached_record_payload(key, {"key": key, "blob": "z" * 32})

    # Old filler under budget pressure.
    for index in range(3):
        filler = cache_dir / f"old-{index}.pickle"
        filler.write_bytes(b"f" * 64)
        os.utime(filler, ns=(1_500_000_000_000_000_000 + index, 1_500_000_000_000_000_000 + index))

    _trim_pickle_cache_dir(cache_dir, max_files=1000, max_bytes=1, force=True)

    for key in run_keys:
        assert (cache_dir / f"{key}.pickle").is_file(), key

    end_analysis_cache_run(force_trim=False)


def _write_gray(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 32), int(value), dtype=np.uint8)
    try:
        import cv2

        assert cv2.imwrite(str(path), image)
    except Exception:
        from karakal.core.image_io import _grayscale_array_to_qimage

        assert _grayscale_array_to_qimage(image).save(str(path), "PNG")


def test_over_budget_cache_run_still_aggregates_confidence(tmp_path, monkeypatch, isolated_analysis_cache) -> None:
    """Even when disk cache is tiny, in-memory aggregation must score every frame."""

    class _Perf:
        disk_cache_limit_mb = 0
        ram_cache_limit_mb = 64
        sequential_debug_mode = True
        parallel_enabled = False
        cpu_workers = 1

    monkeypatch.setattr(cache_mod, "_active_performance_config", lambda: _Perf())
    monkeypatch.setattr("karakal.core.image_io._active_performance_config", lambda: _Perf())
    monkeypatch.setattr("karakal.core.repository_shared._active_performance_config", lambda: _Perf())

    root = tmp_path / "frames"
    records: list[FrameRecord] = []
    model_a = "result_direct"
    model_b = "result_inv"
    for index in range(24):
        key = f"frame_{index:04d}"
        mask_a = root / model_a / f"{key}.png"
        mask_b = root / model_b / f"{key}.png"
        prob_a = root / f"{model_a}_conf" / f"{key}.png"
        prob_b = root / f"{model_b}_conf" / f"{key}.png"
        _write_gray(mask_a, 0 if index % 2 == 0 else 255)
        _write_gray(mask_b, 255 if index % 3 == 0 else 0)
        _write_gray(prob_a, 40 + (index % 50))
        _write_gray(prob_b, 90 + (index % 40))
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
            threshold=0.5,
        ),
        ModelSpec(
            model_id=model_b,
            display_name=model_b,
            mask_folder=root / model_b,
            prob_folder=root / f"{model_b}_conf",
            threshold=0.5,
        ),
    )
    options = BuildOptions(
        geometry_mode=GeometryMode.MASK,
        analysis_max_side=64,
        max_workers=1,
        cache_enabled=True,
        point_match_radius=3.0,
    )
    build = BuildResult(
        records=tuple(records),
        model_specs=specs,
        options=options,
        available_metric_keys=(
            f"model_output_confidence::{model_a}",
            f"model_output_confidence::{model_b}",
        ),
        selected_metric_key=f"model_output_confidence::{model_a}",
    )

    scored = compute_build_result_analytics(
        build,
        metric_key=f"model_output_confidence::{model_a}",
    )
    ready = [record for record in scored.records if bool(getattr(record, "score_ready", False))]
    assert len(ready) == len(records)
    assert all(record.score_percentile is not None for record in ready)

    scored_b = compute_build_result_analytics(
        replace(scored, selected_metric_key=f"model_output_confidence::{model_b}"),
        metric_key=f"model_output_confidence::{model_b}",
    )
    ready_b = [record for record in scored_b.records if bool(getattr(record, "score_ready", False))]
    assert len(ready_b) == len(records)
