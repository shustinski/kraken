"""ThreadPool vs ProcessPool bench for analytics (paths/indices only to workers).

Run:
  pytest plugins/karakal/tests/performance/test_analytics_pool_bench.py -q -s
  set KARAKAL_POOL_BENCH_FRAMES=2000 for the requested 2k comparison.
"""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from karakal.core.analytics import (
    _analyze_record_payload_for_executor,
    _use_thread_pool_for_analytics,
)
from karakal.core.domain import BuildOptions, FrameRecord, GeometryMode, ModelSpec


def _write_gray(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((48, 48), int(value), dtype=np.uint8)
    import cv2

    assert cv2.imwrite(str(path), image)


def _make_records(root: Path, count: int) -> tuple[list[FrameRecord], tuple[ModelSpec, ...]]:
    model_a = "result_direct"
    model_b = "result_inv"
    records: list[FrameRecord] = []
    for index in range(count):
        key = f"frame_{index:05d}"
        mask_a = root / model_a / f"{key}.png"
        mask_b = root / model_b / f"{key}.png"
        prob_a = root / f"{model_a}_conf" / f"{key}.png"
        prob_b = root / f"{model_b}_conf" / f"{key}.png"
        _write_gray(mask_a, (index * 3) % 255)
        _write_gray(mask_b, (index * 7) % 255)
        _write_gray(prob_a, 30 + (index % 100))
        _write_gray(prob_b, 60 + (index % 80))
        records.append(
            FrameRecord(
                key=key,
                display_name=key,
                model_mask_paths={model_a: str(mask_a), model_b: str(mask_b)},
                model_prob_paths={model_a: str(prob_a), model_b: str(prob_b)},
            )
        )
    specs = (
        ModelSpec(model_id=model_a, display_name=model_a, mask_folder=root / model_a, prob_folder=root / f"{model_a}_conf"),
        ModelSpec(model_id=model_b, display_name=model_b, mask_folder=root / model_b, prob_folder=root / f"{model_b}_conf"),
    )
    return records, specs


def _run_pool(records, specs, *, use_threads: bool, workers: int) -> float:
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

    work = [
        (
            record,
            specs,
            64,
            GeometryMode.MASK,
            3.0,
            1,
            0.1,
            3,
            "weighted",
            False,
            False,
            True,
            False,
            False,
            (),
            False,
        )
        for record in records
    ]
    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    started = perf_counter()
    with executor_cls(max_workers=workers) as executor:
        futures = [executor.submit(_analyze_record_payload_for_executor, item) for item in work]
        done = 0
        for future in as_completed(futures):
            key, payload = future.result()
            assert key
            assert payload is not None
            done += 1
        assert done == len(records)
    return perf_counter() - started


@pytest.mark.performance
def test_thread_vs_process_pool_bench(tmp_path, monkeypatch) -> None:
    frames = int(os.environ.get("KARAKAL_POOL_BENCH_FRAMES", "64"))
    workers = int(os.environ.get("KARAKAL_POOL_BENCH_WORKERS", "4"))
    records, specs = _make_records(tmp_path, frames)

    # Warm one sequential call so imports/caches settle.
    _analyze_record_payload_for_executor(
        (
            records[0],
            specs,
            64,
            GeometryMode.MASK,
            3.0,
            1,
            0.1,
            3,
            "weighted",
            False,
            False,
            True,
            False,
            False,
            (),
            False,
        )
    )

    thread_s = _run_pool(records, specs, use_threads=True, workers=workers)
    process_s = _run_pool(records, specs, use_threads=False, workers=workers)
    print(
        f"\npool_bench frames={frames} workers={workers} "
        f"threads={thread_s:.3f}s ({1000 * thread_s / frames:.2f} ms/frame) "
        f"processes={process_s:.3f}s ({1000 * process_s / frames:.2f} ms/frame) "
        f"default_use_threads={_use_thread_pool_for_analytics()}"
    )
    assert thread_s > 0 and process_s > 0
