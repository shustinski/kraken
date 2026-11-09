from __future__ import annotations

import cProfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    from neuralimage.infrastructure import profiling, runtime_config
    for var in (
        "NEURALIMAGE_PROFILE", "NEURALIMAGE_PROFILING", "NEURALIMAGE_PROFILE_ALL",
        "NEURALIMAGE_PROFILE_TRAINING", "NEURALIMAGE_PROFILE_RECOGNITION",
        "NEURALIMAGE_PROFILE_TOP", "NEURALIMAGE_PROFILE_TRAINING_TOP",
        "NEURALIMAGE_PROFILE_RECOGNITION_TOP", "NEURALIMAGE_PROFILE_TRAINING_STEP_STRIDE",
        "NEURALIMAGE_CONFIG",
    ):
        monkeypatch.delenv(var, raising=False)
    runtime_config.clear_runtime_config_cache()
    profiling.reset_profile_output()
    profiling.set_profile_base_dir(tmp_path)
    yield
    profiling.reset_profile_output()
    profiling.set_profile_base_dir(None)
    runtime_config.clear_runtime_config_cache()


def test_enabled_default_false():
    from neuralimage.infrastructure import profiling
    assert profiling.training_profiling_enabled() is False
    assert profiling.recognition_profiling_enabled() is False


def test_enabled_env_per_kind(monkeypatch):
    from neuralimage.infrastructure import profiling
    monkeypatch.setenv("NEURALIMAGE_PROFILE_TRAINING", "1")
    assert profiling.training_profiling_enabled() is True
    assert profiling.recognition_profiling_enabled() is False


def test_enabled_env_master(monkeypatch):
    from neuralimage.infrastructure import profiling
    monkeypatch.setenv("NEURALIMAGE_PROFILE", "on")
    assert profiling.training_profiling_enabled() is True
    assert profiling.recognition_profiling_enabled() is True


def test_enabled_env_beats_ini(monkeypatch, tmp_path):
    from neuralimage.infrastructure import profiling, runtime_config
    ini = tmp_path / "n.ini"
    ini.write_text("[profiling]\nenabled = true\ntraining = false\n", encoding="utf-8")
    monkeypatch.setenv("NEURALIMAGE_CONFIG", str(ini))
    runtime_config.clear_runtime_config_cache()
    monkeypatch.setenv("NEURALIMAGE_PROFILE_TRAINING", "1")
    # env per-kind must win even though ini says training=false
    assert profiling.training_profiling_enabled() is True


def test_enabled_inherits_ini_master(monkeypatch, tmp_path):
    from neuralimage.infrastructure import profiling, runtime_config
    ini = tmp_path / "n.ini"
    ini.write_text("[profiling]\nenabled = true\n", encoding="utf-8")
    monkeypatch.setenv("NEURALIMAGE_CONFIG", str(ini))
    runtime_config.clear_runtime_config_cache()
    assert profiling.training_profiling_enabled() is True
    assert profiling.recognition_profiling_enabled() is True


def test_try_enable_conflict_returns_false():
    from neuralimage.infrastructure import profiling
    profiler = cProfile.Profile()
    try:
        profiler.enable()
    except Exception:
        pass
    # force a conflict by leaving it enabled
    try:
        assert profiling.try_enable_profiler(cProfile.Profile()) is False
    finally:
        profiler.disable()


def test_write_report_creates_pid_suffixed_file(tmp_path):
    from neuralimage.infrastructure import profiling
    profiling.set_profile_base_dir(tmp_path)
    profiling.reset_profile_output()
    profiling.write_profile_report("hello-report")
    files = list(tmp_path.glob("profiling.log-*"))
    assert files, "expected a pid+timestamp suffixed log file"
    assert "hello-report" in files[0].read_text(encoding="utf-8")


def test_prune_profile_logs(monkeypatch, tmp_path):
    from neuralimage.infrastructure import profiling
    monkeypatch.setattr(profiling, "MAX_PROFILE_LOG_FILES", 3)
    configured = tmp_path / "profiling.log"
    for i in range(5):
        p = tmp_path / f"profiling.log-{i}-p1.log"
        p.write_text("x", encoding="utf-8")
        p.touch()
    profiling._prune_profile_logs(configured)
    remaining = list(tmp_path.glob("profiling.log-*"))
    assert len(remaining) == 3  # newest 3 retained, oldest 2 removed


def test_cprofile_region_disabled_noop(monkeypatch):
    from neuralimage.infrastructure import profiling
    calls = []
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    with profiling.cprofile_region("recognition", label="recognition frame", extra="x"):
        pass
    assert calls == []


def test_cprofile_region_enabled_writes_report(monkeypatch):
    from neuralimage.infrastructure import profiling
    calls = []
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    monkeypatch.setattr(profiling, "profiling_enabled", lambda *a, **k: True)
    with profiling.cprofile_region("recognition", label="recognition frame", extra="frame=1/1 name=a.png"):
        sum(range(1000))
    assert calls
    output = "\n".join(str(m) for c in calls for m in c)
    assert "[neuralimage recognition frame]" in output
    assert "wall_ms=" in output
    assert "[neuralimage recognition frame stats]" in output


def test_cprofile_region_busy_falls_back_wallclock(monkeypatch):
    from neuralimage.infrastructure import profiling
    calls = []
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    monkeypatch.setattr(profiling, "profiling_enabled", lambda *a, **k: True)
    monkeypatch.setattr(profiling, "try_enable_profiler", lambda p: False)
    with profiling.cprofile_region("recognition", label="recognition frame", extra="x"):
        pass
    output = "\n".join(str(m) for c in calls for m in c)
    assert "wall_ms=" in output
    assert "cprofile_skipped=yes" in output


def test_training_profile_stride_default(monkeypatch):
    from neuralimage.infrastructure import profiling
    assert profiling.training_profile_stride() == 1
