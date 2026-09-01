"""CLI benchmark for Karakal inter-model comparison."""
from __future__ import annotations

import argparse
import json
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from .cache import ComparisonResultCache
from .ensemble import compare_ensemble
from .models import EnsembleComparisonRequest, ModelFrameResult, PairwiseComparisonRequest
from .pairwise import compare_pairwise


@dataclass(slots=True)
class BenchmarkStage:
    name: str
    duration_ms: float


@dataclass(slots=True)
class BenchmarkCaseResult:
    case_name: str
    mode: str
    profile: str
    model_count: int
    shape: tuple[int, int]
    cache_state: str
    total_time_ms: float
    peak_memory_mb: float
    stages: list[BenchmarkStage] = field(default_factory=list)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Karakal inter-model comparison.")
    parser.add_argument("--output-dir", type=Path, default=Path("build") / "karakal-comparison-benchmark")
    parser.add_argument("--repeat", type=int, default=3, help="Measurements per case; values below three are clamped.")
    parser.add_argument("--sizes", default="small,medium,large")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_sizes = {item.strip().lower() for item in str(args.sizes).split(",") if item.strip()}

    results: list[BenchmarkCaseResult] = []
    for _ in range(max(3, int(args.repeat))):
        for case_name, shape in _case_shapes().items():
            if case_name not in selected_sizes:
                continue
            results.extend(_run_case(case_name, shape))

    payload = _results_payload(results)
    baseline_path = output_dir / "baseline.json"
    final_path = output_dir / "latest.json"
    baseline_payload = None
    if baseline_path.is_file():
        try:
            baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception:
            baseline_payload = None
    if not baseline_path.is_file():
        baseline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    final_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = _format_report(results, baseline_payload=baseline_payload)
    (output_dir / "latest.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved benchmark report: {final_path}")
    return 0


def _case_shapes() -> dict[str, tuple[int, int]]:
    return {
        "small": (64, 64),
        "medium": (256, 256),
        "large": (768, 768),
    }


def _run_case(case_name: str, shape: tuple[int, int]) -> list[BenchmarkCaseResult]:
    pair_models = _synthetic_models("pair", shape, 2)
    ensemble4 = _synthetic_models("ens4", shape, 4)
    ensemble8 = _synthetic_models("ens8", shape, 8)
    result_cache = ComparisonResultCache(max_items=64)
    rows: list[BenchmarkCaseResult] = []
    rows.append(_measure_pairwise(case_name, shape, pair_models, "polygon", "cold", cache=None))
    rows.append(_measure_pairwise(case_name, shape, pair_models, "polygon", "warm", cache=result_cache))
    rows.append(_measure_pairwise(case_name, shape, pair_models, "line_network", "cold", cache=None))
    rows.append(_measure_pairwise(case_name, shape, pair_models, "line_network", "warm", cache=result_cache))
    rows.append(_measure_ensemble(case_name, shape, ensemble4, "polygon", "cold", cache=None))
    rows.append(_measure_ensemble(case_name, shape, ensemble4, "polygon", "warm", cache=result_cache))
    rows.append(_measure_ensemble(case_name, shape, ensemble8, "line_network", "cold", cache=None))
    rows.append(_measure_ensemble(case_name, shape, ensemble8, "line_network", "warm", cache=result_cache))
    return rows


def _synthetic_models(prefix: str, shape: tuple[int, int], count: int) -> tuple[ModelFrameResult, ...]:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    center_y = height / 2.0
    center_x = width / 2.0
    radius = max(4.0, min(height, width) * 0.22)
    rows: list[ModelFrameResult] = []
    for index in range(count):
        shift = index - count // 2
        circle = (yy - center_y - shift) ** 2 + (xx - center_x + shift) ** 2 <= radius**2
        line = np.zeros(shape, dtype=bool)
        line[max(0, height // 3 + shift) : min(height, height // 3 + shift + 2), width // 5 : width - width // 5] = True
        mask = np.logical_or(circle, line)
        probability = np.clip(mask.astype(np.float32) * 0.75 + 0.15 + (xx.astype(np.float32) / max(1, width)) * 0.05, 0.0, 1.0)
        rows.append(ModelFrameResult(f"{prefix}_{index}", f"{case_frame_id(prefix, shape)}", probability, mask, {"geometry_mode": "polygon"}))
    return tuple(rows)


def case_frame_id(prefix: str, shape: tuple[int, int]) -> str:
    return f"{prefix}_{shape[0]}x{shape[1]}"


def _measure_pairwise(
    case_name: str,
    shape: tuple[int, int],
    models: tuple[ModelFrameResult, ...],
    profile: str,
    cache_state: str,
    cache: ComparisonResultCache | None,
) -> BenchmarkCaseResult:
    request = PairwiseComparisonRequest(case_frame_id(f"pair_{profile}", shape), models[0], models[1], profile=profile)
    return _measure(
        case_name=case_name,
        shape=shape,
        mode="pairwise",
        profile=profile,
        model_count=2,
        cache_state=cache_state,
        cache=cache,
        action=lambda: compare_pairwise(request, result_cache=cache).frame,
    )


def _measure_ensemble(
    case_name: str,
    shape: tuple[int, int],
    models: tuple[ModelFrameResult, ...],
    profile: str,
    cache_state: str,
    cache: ComparisonResultCache | None,
) -> BenchmarkCaseResult:
    request = EnsembleComparisonRequest(case_frame_id(f"ensemble_{profile}_{len(models)}", shape), models, profile=profile)
    return _measure(
        case_name=case_name,
        shape=shape,
        mode="ensemble",
        profile=profile,
        model_count=len(models),
        cache_state=cache_state,
        cache=cache,
        action=lambda: compare_ensemble(request, result_cache=cache).frame,
    )


def _measure(
    *,
    case_name: str,
    shape: tuple[int, int],
    mode: str,
    profile: str,
    model_count: int,
    cache_state: str,
    cache: ComparisonResultCache | None,
    action: Callable[[], object],
) -> BenchmarkCaseResult:
    if cache is not None and cache_state == "warm":
        action()
    tracemalloc.start()
    started = perf_counter()
    result = action()
    total = 1000.0 * (perf_counter() - started)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stage_timings = dict(getattr(result, "metadata", {}).get("stage_timings_ms", {}) or {})
    stages = [BenchmarkStage(name=key, duration_ms=float(value)) for key, value in sorted(stage_timings.items())]
    return BenchmarkCaseResult(
        case_name=case_name,
        mode=mode,
        profile=profile,
        model_count=model_count,
        shape=shape,
        cache_state=cache_state,
        total_time_ms=float(total),
        peak_memory_mb=float(peak / (1024.0 * 1024.0)),
        stages=stages,
    )


def _results_payload(results: list[BenchmarkCaseResult]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "results": [
            {
                "case_name": row.case_name,
                "mode": row.mode,
                "profile": row.profile,
                "model_count": row.model_count,
                "shape": row.shape,
                "cache_state": row.cache_state,
                "total_time_ms": row.total_time_ms,
                "peak_memory_mb": row.peak_memory_mb,
                "stages": {stage.name: stage.duration_ms for stage in row.stages},
            }
            for row in results
        ],
    }


def _format_report(results: list[BenchmarkCaseResult], *, baseline_payload: dict[str, object] | None = None) -> str:
    lines = [
        "Karakal comparison benchmark",
        "case | mode | profile | K | shape | cache | total_ms | peak_mb | slowest_stage",
    ]
    for row in results:
        slowest = max(row.stages, key=lambda item: item.duration_ms, default=BenchmarkStage("-", 0.0))
        lines.append(
            f"{row.case_name} | {row.mode} | {row.profile} | {row.model_count} | "
            f"{row.shape[0]}x{row.shape[1]} | {row.cache_state} | "
            f"{row.total_time_ms:.2f} | {row.peak_memory_mb:.2f} | {slowest.name}:{slowest.duration_ms:.2f}"
        )
    if baseline_payload:
        baseline_rows = {
            _result_key(row): row
            for row in baseline_payload.get("results", [])
            if isinstance(row, dict)
        }
        if baseline_rows:
            lines.extend(["", "baseline comparison", "case | mode | profile | K | cache | baseline_ms | latest_ms | delta_ms | speedup"])
            for row in results:
                baseline = baseline_rows.get(_case_result_key(row))
                if not isinstance(baseline, dict):
                    continue
                baseline_ms = float(baseline.get("total_time_ms") or 0.0)
                if baseline_ms <= 0.0:
                    continue
                delta = float(row.total_time_ms - baseline_ms)
                speedup = baseline_ms / max(1e-9, float(row.total_time_ms))
                lines.append(
                    f"{row.case_name} | {row.mode} | {row.profile} | {row.model_count} | {row.cache_state} | "
                    f"{baseline_ms:.2f} | {row.total_time_ms:.2f} | {delta:+.2f} | {speedup:.2f}x"
                )
    return "\n".join(lines)


def _case_result_key(row: BenchmarkCaseResult) -> tuple[object, ...]:
    return (row.case_name, row.mode, row.profile, row.model_count, tuple(row.shape), row.cache_state)


def _result_key(row: dict[str, object]) -> tuple[object, ...]:
    shape = row.get("shape") or ()
    return (
        str(row.get("case_name") or ""),
        str(row.get("mode") or ""),
        str(row.get("profile") or ""),
        int(row.get("model_count") or 0),
        tuple(shape) if isinstance(shape, (list, tuple)) else (),
        str(row.get("cache_state") or ""),
    )


if __name__ == "__main__":
    raise SystemExit(main())
