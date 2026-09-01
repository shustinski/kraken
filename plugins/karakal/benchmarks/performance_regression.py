"""Compare Karakal benchmark JSON files with configurable regression thresholds."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RegressionComparison:
    baseline_seconds: float
    candidate_seconds: float
    percent_change: float
    speedup: float
    status: str
    peak_ram_change_percent: float | None


def _duration_seconds(payload: dict[str, object]) -> float:
    if "wall_time_seconds" in payload:
        return float(payload["wall_time_seconds"])
    if "elapsed_ms" in payload:
        return float(payload["elapsed_ms"]) / 1000.0
    raise ValueError("Benchmark payload has neither wall_time_seconds nor elapsed_ms")


def _peak_ram(payload: dict[str, object]) -> float | None:
    value = payload.get("peak_rss_mb")
    if value is None:
        environment = payload.get("environment")
        if isinstance(environment, dict):
            value = environment.get("peak_ram_mb")
    return None if value is None else float(value)


def compare_payloads(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    warning_threshold: float = 0.10,
    failure_threshold: float = 0.25,
) -> RegressionComparison:
    baseline_seconds = _duration_seconds(baseline)
    candidate_seconds = _duration_seconds(candidate)
    if baseline_seconds <= 0.0:
        raise ValueError("Baseline duration must be positive")
    relative_change = candidate_seconds / baseline_seconds - 1.0
    status = "failure" if relative_change > failure_threshold else "warning" if relative_change > warning_threshold else "ok"
    baseline_ram = _peak_ram(baseline)
    candidate_ram = _peak_ram(candidate)
    ram_change = None
    if baseline_ram is not None and candidate_ram is not None and baseline_ram > 0.0:
        ram_change = 100.0 * (candidate_ram / baseline_ram - 1.0)
    return RegressionComparison(
        baseline_seconds=baseline_seconds,
        candidate_seconds=candidate_seconds,
        percent_change=100.0 * relative_change,
        speedup=baseline_seconds / max(1e-12, candidate_seconds),
        status=status,
        peak_ram_change_percent=ram_change,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--warning", type=float, default=0.10)
    parser.add_argument("--failure", type=float, default=0.25)
    args = parser.parse_args(argv)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise SystemExit("Benchmark files must contain JSON objects")
    comparison = compare_payloads(
        baseline,
        candidate,
        warning_threshold=max(0.0, float(args.warning)),
        failure_threshold=max(0.0, float(args.failure)),
    )
    print(json.dumps({
        "status": comparison.status,
        "baseline_seconds": comparison.baseline_seconds,
        "candidate_seconds": comparison.candidate_seconds,
        "percent_change": comparison.percent_change,
        "speedup": comparison.speedup,
        "peak_ram_change_percent": comparison.peak_ram_change_percent,
    }, indent=2))
    return 2 if comparison.status == "failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
