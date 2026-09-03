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
    load_runtime_config,
    runtime_config_path,
)


@pytest.fixture(autouse=True)
def _reset_runtime_config_after_test():
    yield
    profiling.reset_profile_output()
    clear_runtime_config_cache()


def _use_config(monkeypatch, path: Path) -> None:
    monkeypatch.setenv(CONFIG_ENV, str(path))
    profiling.reset_profile_output()
    clear_runtime_config_cache()


def test_shipped_config_disables_profiling_and_contains_runtime_sections(monkeypatch) -> None:
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    clear_runtime_config_cache()

    parser = load_runtime_config()

    assert bundled_config_path().is_file()
    assert not parser.getboolean("profiling", "enabled")
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


def test_profile_report_is_written_to_configured_log(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "profile.log"
    config_path = tmp_path / "custom.ini"
    config_path.write_text(
        f"[profiling]\nlog_file = {log_path}\nmax_log_bytes = 4096\nbackup_count = 1\n",
        encoding="utf-8",
    )
    _use_config(monkeypatch, config_path)

    profiling.write_profile_report("[contour test profiling] summary", "stats")
    profiling.reset_profile_output()

    output = log_path.read_text(encoding="utf-8")
    assert "[contour test profiling] summary" in output
    assert "stats" in output
