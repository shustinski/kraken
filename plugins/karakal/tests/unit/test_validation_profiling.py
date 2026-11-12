"""The validation profiler sees the work of parallel analytics threads, not only the calling thread."""

from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from karakal.core import analytics
from karakal.core.analytics import compute_build_result_analytics
from karakal.core.domain import BuildOptions, BuildResult, FrameRecord, ModelSpec
from karakal.core.performance import PerformanceConfig, ProfilingMode
from karakal.core.profiling import ProfilerRun, activate_profiler
from karakal.core.synthetic_validation import make_case

FRAMES = 6


def _write(path: Path, image) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return str(path)


def _build_result(tmp_path: Path, model_ids: tuple[str, ...]) -> BuildResult:
    records = []
    for index in range(FRAMES):
        frame = make_case(("clean", "open", "bridge")[index % 3], index)
        key = f"F_{index:02d}"
        records.append(
            FrameRecord(
                key,
                key,
                original_path=_write(tmp_path / "photos" / f"{key}.png", frame.photo),
                model_mask_paths={m: _write(tmp_path / m / f"{key}.png", frame.masks[m]) for m in model_ids},
                model_prob_paths={m: _write(tmp_path / m / f"{key}_confidence.png", frame.confidences[m]) for m in model_ids},
            )
        )
    specs = tuple(ModelSpec(m, m, tmp_path / m, prob_folder=tmp_path / m) for m in model_ids)
    return BuildResult(records=tuple(records), model_specs=specs, options=BuildOptions(cache_enabled=False, max_workers=3))


def _stage_calls(profiler: ProfilerRun, name: str) -> int:
    return sum(int(stage["calls"]) for stage in profiler.snapshot().stages if stage["name"] == name)


def test_model_comparison_stages_are_counted_in_pool_threads(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(analytics, "_use_thread_pool_for_analytics", lambda: True)
    monkeypatch.setattr(analytics, "_analytics_worker_count", lambda *args, **kwargs: 3)
    profiler = ProfilerRun("test", PerformanceConfig(profiling_mode=ProfilingMode.DETAILED))
    with activate_profiler(profiler):
        result = compute_build_result_analytics(_build_result(tmp_path, ("a", "b")), metric_key="overall_frame_score")
    assert all(record.score_ready for record in result.records)
    assert _stage_calls(profiler, "validation.frame.loading_preprocess") == FRAMES
    assert _stage_calls(profiler, "validation.metrics.pairwise") == FRAMES


def test_single_result_risk_stages_are_counted_in_pool_threads(tmp_path: Path) -> None:
    profiler = ProfilerRun("test", PerformanceConfig(profiling_mode=ProfilingMode.DETAILED))
    with activate_profiler(profiler):
        result = compute_build_result_analytics(_build_result(tmp_path, ("a",)), metric_key="single_result_risk_score")
    assert all(record.score_ready for record in result.records)
    for stage in ("validation.risk.mask_structure", "validation.risk.source_alignment", "validation.risk.source_mask"):
        assert _stage_calls(profiler, stage) == FRAMES
    assert _stage_calls(profiler, "validation.risk.confidence") == FRAMES


def test_profiling_off_records_nothing(tmp_path: Path) -> None:
    profiler = ProfilerRun("test", PerformanceConfig(profiling_mode=ProfilingMode.OFF))
    with activate_profiler(profiler):
        compute_build_result_analytics(_build_result(tmp_path, ("a",)), metric_key="single_result_risk_score")
    assert profiler.snapshot().stages == ()
