from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.performance


def _module():
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "performance_regression.py"
    spec = importlib.util.spec_from_file_location("karakal_performance_regression", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_regression_thresholds_distinguish_warning_and_failure() -> None:
    module = _module()
    baseline = {"wall_time_seconds": 10.0, "peak_rss_mb": 100.0}

    warning = module.compare_payloads(baseline, {"wall_time_seconds": 11.5, "peak_rss_mb": 105.0})
    failure = module.compare_payloads(baseline, {"wall_time_seconds": 13.0, "peak_rss_mb": 120.0})

    assert warning.status == "warning"
    assert warning.peak_ram_change_percent == pytest.approx(5.0)
    assert failure.status == "failure"


def test_faster_candidate_reports_speedup() -> None:
    comparison = _module().compare_payloads(
        {"elapsed_ms": 2000.0},
        {"elapsed_ms": 1000.0},
    )

    assert comparison.status == "ok"
    assert comparison.speedup == pytest.approx(2.0)
    assert comparison.percent_change == pytest.approx(-50.0)
