from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent

from contour.graphics.editor_hotkeys import event_matches_shortcut, shortcut_native_text, tool_shortcut_native_text
from contour.graphics.tools import EditorTool
from contour.infrastructure import profiling
from contour.infrastructure.runtime_config import (
    CONFIG_ENV,
    bundled_config_path,
    clear_runtime_config_cache,
    contour_application_directory,
    load_runtime_config,
    profiling_log_path,
    runtime_config_path,
)


@pytest.fixture(autouse=True)
def _reset_runtime_config_after_test():
    yield
    profiling.reset_profile_output()
    clear_runtime_config_cache()


def _use_config(monkeypatch, path: Path) -> None:
    monkeypatch.setenv(CONFIG_ENV, str(path))
    for env_name in ("CONTOUR_PROFILE", "CONTOUR_PROFILING", "CONTOUR_PROFILE_ALL"):
        monkeypatch.delenv(env_name, raising=False)
    profiling.reset_profile_output()
    clear_runtime_config_cache()


def test_shipped_config_enables_startup_profiling_and_contains_runtime_sections(monkeypatch) -> None:
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    clear_runtime_config_cache()

    parser = load_runtime_config()

    assert bundled_config_path().is_file()
    assert parser.getboolean("profiling", "startup")
    assert parser.get("profiling", "frame_switch") == ""
    assert parser.has_section("shortcuts")
    assert parser.has_section("editor")
    assert parser.has_section("large_dataset")


def test_ini_controls_profiler_shortcuts_and_top_lines(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "custom.ini"
    config_path.write_text(
        "[profiling]\n"
        "enabled = false\n"
        "frame_switch = true\n"
        "frame_switch_top_lines = 17\n"
        "[shortcuts]\n"
        "fit_view = Ctrl+F\n"
        "tool_select = Q\n",
        encoding="utf-8",
    )
    _use_config(monkeypatch, config_path)

    assert runtime_config_path() == config_path
    assert profiling.frame_switch_profiling_enabled()
    assert not profiling.processing_profiling_enabled()
    assert profiling.frame_switch_top_lines() == 17
    assert shortcut_native_text("fit_view", "F") == "Ctrl+F"
    assert tool_shortcut_native_text(EditorTool.SELECT) == "Q"
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_F, Qt.KeyboardModifier.ControlModifier)
    assert event_matches_shortcut(event, "fit_view", "F")


def test_environment_override_has_priority_over_ini(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "custom.ini"
    config_path.write_text("[profiling]\nframe_switch = false\n", encoding="utf-8")
    _use_config(monkeypatch, config_path)
    monkeypatch.setenv("CONTOUR_PROFILE_FRAME_SWITCH", "1")

    assert profiling.frame_switch_profiling_enabled()


def test_startup_profiler_uses_its_own_switch_and_top_lines(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "custom.ini"
    config_path.write_text(
        "[profiling]\n"
        "enabled = false\n"
        "startup = true\n"
        "startup_top_lines = 19\n",
        encoding="utf-8",
    )
    _use_config(monkeypatch, config_path)

    assert profiling.startup_profiling_enabled()
    assert profiling.startup_top_lines() == 19


def test_profile_report_is_written_to_configured_log(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "profile.log"
    config_path = tmp_path / "custom.ini"
    config_path.write_text(
        f"[profiling]\nlog_file = {log_path}\n",
        encoding="utf-8",
    )
    _use_config(monkeypatch, config_path)

    profiling.write_profile_report("[contour test profiling] summary", "stats")
    profiling.reset_profile_output()
    profiling.write_profile_report("[contour test profiling] second run")
    profiling.reset_profile_output()

    generated_logs = list(log_path.parent.glob("profile-*-p*.log"))
    assert len(generated_logs) == 2
    output = "\n".join(path.read_text(encoding="utf-8") for path in generated_logs)
    assert "[contour test profiling] summary" in output
    assert "[contour test profiling] second run" in output
    assert "stats" in output


def test_profile_log_retention_keeps_newest_files(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "profiling.log"
    config_path = tmp_path / "custom.ini"
    config_path.write_text(f"[profiling]\nlog_file = {log_path}\n", encoding="utf-8")
    _use_config(monkeypatch, config_path)
    assert profiling.MAX_PROFILE_LOG_FILES == 500
    monkeypatch.setattr(profiling, "MAX_PROFILE_LOG_FILES", 3)
    for index in range(3):
        (tmp_path / f"profiling-20200101-000000-{index:06d}-p1.log").write_text("old", encoding="utf-8")

    profiling.write_profile_report("new run")
    profiling.reset_profile_output()

    generated_logs = list(tmp_path.glob("profiling-*-p*.log"))
    assert len(generated_logs) == 3
    assert any("new run" in path.read_text(encoding="utf-8") for path in generated_logs)


def test_relative_profiling_log_is_in_logs_next_to_frozen_executable(tmp_path, monkeypatch) -> None:
    executable = tmp_path / "Contour" / "Contour.exe"
    config_path = tmp_path / "custom.ini"
    config_path.write_text("[profiling]\nlog_file = profiling.log\n", encoding="utf-8")
    _use_config(monkeypatch, config_path)
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys.executable", str(executable))

    assert contour_application_directory() == executable.parent
    assert profiling_log_path() == executable.parent / "logs" / "profiling.log"


def test_absolute_profiling_log_path_is_preserved(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "diagnostics" / "startup.log"
    config_path = tmp_path / "custom.ini"
    config_path.write_text(f"[profiling]\nlog_file = {log_path}\n", encoding="utf-8")
    _use_config(monkeypatch, config_path)

    assert profiling_log_path() == log_path
