from __future__ import annotations

import json

import numpy as np

from karakal.core.performance import PerformanceConfig, ProfilingMode, load_performance_config
from karakal.core.profiling import ProfilerRun, activate_profiler, current_profiler, export_profile, profile_stage
from karakal.core.grid_anomaly import detect_grid_cell_anomalies
from karakal.ui.profiling_dialog import ProfilingDialog


def test_performance_config_environment_precedence_and_legacy_aliases() -> None:
    config = load_performance_config(
        {"cpu_workers": 2, "batch_size": 3, "profiling_mode": "summary"},
        {
            "KARAKAL_GRID_INSPECTION_WORKERS": "4",
            "KARAKAL_BATCH_SIZE": "12",
            "KARAKAL_PROFILING_MODE": "detailed",
        },
    )

    assert 1 <= config.cpu_workers <= 4
    assert config.batch_size == 12
    assert config.profiling_mode is ProfilingMode.DETAILED


def test_disabled_profiler_is_a_noop() -> None:
    profiler = ProfilerRun("validation", PerformanceConfig(profiling_mode=ProfilingMode.OFF))

    with profiler.stage("validation.metrics", frame_count=1):
        profiler.increment("frames.processed")

    snapshot = profiler.snapshot()
    assert snapshot.stages == ()
    assert snapshot.counters == {}


def test_nested_profiler_records_inclusive_and_self_time() -> None:
    profiler = ProfilerRun("validation", PerformanceConfig(profiling_mode=ProfilingMode.SUMMARY))

    with profiler.stage("validation", frame_count=1):
        with profiler.stage("validation.metrics.dice"):
            sum(range(100))
        profiler.increment("frames.processed")

    rows = {str(row["name"]): row for row in profiler.snapshot().stages}
    assert rows["validation"]["total_ms"] >= rows["validation.metrics.dice"]["total_ms"]
    assert rows["validation"]["self_ms"] <= rows["validation"]["total_ms"]
    assert rows["validation"]["frame_count"] == 1


def test_profiler_accepts_event_loop_duration_samples() -> None:
    profiler = ProfilerRun("validation", PerformanceConfig(profiling_mode=ProfilingMode.SUMMARY))

    profiler.record_duration("ui.matrix.refresh", 2_000_000, frame_count=4)

    row = profiler.snapshot().stages[0]
    assert row["name"] == "ui.matrix.refresh"
    assert row["total_ms"] == 2.0
    assert row["self_ms"] == 2.0
    assert row["frame_count"] == 4


def test_active_profiler_stage_helper_restores_previous_context() -> None:
    profiler = ProfilerRun("grid", PerformanceConfig(profiling_mode=ProfilingMode.SUMMARY))
    assert current_profiler() is None

    with activate_profiler(profiler):
        assert current_profiler() is profiler
        with profile_stage("validation.grid.contours", frame_id="frame-1", frame_count=1):
            pass

    assert current_profiler() is None
    assert profiler.snapshot().stages[0]["name"] == "validation.grid.contours"


def test_trace_is_bounded_by_frame_limit() -> None:
    config = PerformanceConfig(profiling_mode=ProfilingMode.TRACE, profiling_trace_frame_limit=2)
    profiler = ProfilerRun("validation", config)

    for frame_id in ("one", "two", "three"):
        with profiler.stage("validation.frame", frame_id=frame_id, frame_count=1):
            pass

    begin_events = [event for event in profiler.snapshot().trace_events if event["ph"] == "B"]
    assert {event["args"]["frame_id"] for event in begin_events} == {"one", "two"}


def test_profile_export_is_atomic_and_machine_readable(tmp_path) -> None:
    config = PerformanceConfig(
        profiling_mode=ProfilingMode.TRACE,
        profiling_output_directory=str(tmp_path),
        profiling_trace_enabled=True,
    )
    profiler = ProfilerRun("validation_grid", config, run_id="abc12345")
    with profiler.stage("validation.grid.decode", frame_id="frame", frame_count=1):
        profiler.increment("files.decoded")
        profiler.increment("frames.processed")

    exported = export_profile(profiler.snapshot(), tmp_path, config)

    assert set(exported) == {"json", "csv", "markdown", "trace"}
    payload = json.loads(exported["json"].read_text(encoding="utf-8"))
    assert payload["run_id"] == "abc12345"
    assert payload["counters"]["files.decoded"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_profile_history_retention_groups_all_formats(tmp_path) -> None:
    config = PerformanceConfig(profiling_mode=ProfilingMode.SUMMARY, profiling_keep_last_runs=1)
    first = ProfilerRun("validation", config, run_id="first000")
    export_profile(first.snapshot(), tmp_path, config)
    second = ProfilerRun("validation", config, run_id="second00")
    export_profile(second.snapshot(), tmp_path, config)

    names = {path.name for path in tmp_path.iterdir()}
    assert any("second00" in name for name in names)
    assert not any("first000" in name for name in names)


def test_profiling_does_not_change_grid_result() -> None:
    image = np.zeros((96, 128), dtype=np.uint8)
    image[12:84:16, 12:116:16] = 255
    off = detect_grid_cell_anomalies(image, frame_id="same")
    profiler = ProfilerRun("grid", PerformanceConfig(profiling_mode=ProfilingMode.SUMMARY))

    with activate_profiler(profiler):
        summary = detect_grid_cell_anomalies(image, frame_id="same")

    assert summary == off


def test_profiling_dialog_renders_snapshot_and_changes_future_mode(qtbot) -> None:
    config = PerformanceConfig(profiling_mode=ProfilingMode.SUMMARY)
    profiler = ProfilerRun("validation", config)
    with profiler.stage("validation.metrics.dice", frame_count=2):
        profiler.set_counter("frames.processed", 2)
    dialog = ProfilingDialog(config)
    qtbot.addWidget(dialog)
    changed = []
    dialog.configurationChanged.connect(changed.append)

    dialog.set_snapshot(profiler.snapshot())
    dialog._refresh()
    dialog.mode_combo.setCurrentText("detailed")

    assert dialog.table.rowCount() == 1
    assert dialog.frames_label.text() == "2"
    assert changed[-1].profiling_mode is ProfilingMode.DETAILED
