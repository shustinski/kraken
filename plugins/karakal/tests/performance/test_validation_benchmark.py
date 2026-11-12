"""The validation benchmark runs end to end and writes a report other runs can be compared with."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.performance


def _module():
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark_validation.py"
    spec = importlib.util.spec_from_file_location("karakal_benchmark_validation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("layer", ["conductors", "vias"])
def test_benchmark_writes_every_run_with_stages(tmp_path: Path, layer: str) -> None:
    module = _module()
    output = tmp_path / "out" / "latest.json"
    code = module.main(
        [
            "--sizes", "512",
            "--models", "2,3",
            "--frames", "6",
            "--workers", "1,2",
            "--layer", layer,
            "--data-dir", str(tmp_path / "data"),
            "--output", str(output),
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert output.with_suffix(".md").is_file()
    assert payload["wall_time_seconds"] > 0.0
    keys = [run["key"] for run in payload["runs"]]
    # Model comparison for 2 and 3 models, the four single-model methods once each; two worker counts.
    assert len(keys) == len(set(keys)) == (2 + 4) * 2
    for run in payload["runs"]:
        assert run["frames_scored"] == run["frames"] == 6, run["key"]
        assert run["metric_used"] == run["metric_key"], run["key"]
        assert run["stages"], run["key"]
        if os.name == "nt":
            assert run["peak_rss_mb"] and run["peak_rss_mb"] > 0.0


def test_dataset_is_written_once_and_reused(tmp_path: Path) -> None:
    module = _module()
    first = module.prepare_dataset(tmp_path, 512, 3, 2, "conductors")
    stamp = (first.folder / "photos" / "F_0000.png").stat().st_mtime_ns
    again = module.prepare_dataset(tmp_path, 512, 3, 2, "conductors")
    assert again == first
    assert (first.folder / "photos" / "F_0000.png").stat().st_mtime_ns == stamp


def test_compare_reports_the_change_of_every_run() -> None:
    module = _module()
    baseline = {"runs": [{"key": "k", "wall_time_seconds": 2.0, "ms_per_frame": 20.0}]}
    candidate = {"runs": [{"key": "k", "wall_time_seconds": 1.0, "ms_per_frame": 10.0}, {"key": "n", "wall_time_seconds": 1.0, "ms_per_frame": 1.0}]}
    lines = module.compare_runs(baseline, candidate)
    assert lines == ["k: 20.0 -> 10.0 ms/frame (-50.0%)", "n: new run"]
