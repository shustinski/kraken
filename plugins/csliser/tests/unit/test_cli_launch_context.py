from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication
import pytest

from csliser.actions import create_action_registry
from csliser.application.bootstrap import CSliserApplicationComponents
from csliser.application.cli import main
from csliser.domain.models import FileOperation, FrameRange, SelectionMode
from csliser.domain.ranges import FrameRangeError, expand_frames, parse_frame_ranges
from csliser.infrastructure.settings_store import WindowSettingsStore
from csliser.presentation.qt.window import CSliserWindow
from kraken_core.ipc import ActionRequest


def test_gui_cli_populates_actual_window_without_executing_files(qtbot, tmp_path, monkeypatch):
    from csliser.application import bootstrap
    source = tmp_path / "source"
    source.mkdir()
    (source / "chip_1.jpg").write_bytes(b"preserve original")
    window = CSliserWindow(settings_store=WindowSettingsStore(
        lambda: QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)))
    qtbot.addWidget(window)
    monkeypatch.setattr(bootstrap, "build_application", lambda *args, **kwargs:
        CSliserApplicationComponents(QApplication.instance(), window))
    monkeypatch.setattr(QApplication, "exec", lambda *_args: 0)
    try:
        with pytest.raises(SystemExit) as exited:
            main(["--source", str(source), "--extension", ".jpg", "--frames", "1",
                  "--frames-per-row", "12", "--selection-mode", "full_range",
                  "--operation", "move", "--destination", str(tmp_path / "out")])
        assert exited.value.code == 0
        config = window._config(FileOperation.MOVE)
        assert tuple(s.path for s in config.sources) == (source,)
        assert config.sources[0].extensions == (".jpg",)
        assert config.frame_expression == "1" and config.frames_per_row == 12
        assert config.selection_mode is SelectionMode.FULL_RANGE
        assert config.destination == tmp_path / "out"
        assert window._current_operation is FileOperation.MOVE
        assert window._worker_thread is None and (source / "chip_1.jpg").read_bytes() == b"preserve original"
        assert not (tmp_path / "out").exists()
    finally:
        window.close()


@pytest.mark.parametrize("payload", [{}, {"path": "existing"}])
def test_unbound_legacy_action_does_not_claim_a_window_was_opened(tmp_path, payload):
    if payload:
        payload = {"path": str(tmp_path)}
    response = create_action_registry().dispatch(ActionRequest("open_folder", payload))
    assert not response.ok


def test_huge_selection_is_rejected_before_expansion(monkeypatch):
    from csliser.domain import ranges
    monkeypatch.setattr(ranges, "_unique_in_order", lambda *_args, **_kwargs:
        pytest.fail("Unbounded selection reached expansion"))
    with pytest.raises(FrameRangeError):
        expand_frames((FrameRange(0, 10**12),), mode=SelectionMode.FULL_RANGE, frames_per_row=135)


@pytest.mark.parametrize("expression", [b"1", "9" * 5000, str(2**63), ";".join(["0"] * 4097), "0" * 65537],
                         ids=["bytes", "long-number", "overflow", "range-count", "text-length"])
def test_expression_limits_report_domain_error(expression):
    with pytest.raises(FrameRangeError):
        parse_frame_ranges(expression)


def test_union_limit_and_overlap_work_budget_are_independent():
    with pytest.raises(FrameRangeError, match="unique"):
        expand_frames((FrameRange(0, 2), FrameRange(3, 5)), mode=SelectionMode.FULL_RANGE,
                      frames_per_row=0, max_frames=4)
    with pytest.raises(FrameRangeError, match="visits"):
        expand_frames((FrameRange(0, 2),) * 4, mode=SelectionMode.FULL_RANGE,
                      frames_per_row=0, max_frames=4, max_visits=10)
    assert expand_frames((FrameRange(2, 4), FrameRange(0, 3)), mode=SelectionMode.FULL_RANGE,
                         frames_per_row=0, max_frames=5) == (2, 3, 4, 0, 1)


def test_huge_rectangle_is_rejected_before_iteration(monkeypatch):
    from csliser.domain import ranges
    monkeypatch.setattr(ranges, "_expand_rectangle", lambda *_args:
                        pytest.fail("Oversized rectangle reached iteration"))
    with pytest.raises(FrameRangeError):
        expand_frames((FrameRange(0, 10**12),), mode=SelectionMode.RECTANGLE, frames_per_row=135)


def test_legacy_rectangle_and_zero_based_boundary_remain_supported():
    assert expand_frames(parse_frame_ranges("13:8"), mode=SelectionMode.RECTANGLE,
                         frames_per_row=10) == tuple(range(3, 9)) + tuple(range(13, 19))
    assert parse_frame_ranges("0;9223372036854775807") == (FrameRange(0, 0), FrameRange(2**63-1, 2**63-1))


def test_capabilities_do_not_advertise_unconnected_actions(capsys):
    import json
    main(["--get-capabilities"])
    response = json.loads(capsys.readouterr().out)
    assert response["ok"]
    assert "open_folder" not in str(response)


def test_huge_cli_plan_returns_validation_error_without_traceback(tmp_path, capsys):
    with pytest.raises(SystemExit) as exited:
        main(["--plan", "--source", str(tmp_path), "--destination", str(tmp_path / "out"),
              "--frames", "0-1000000000000", "--selection-mode", "full_range"])
    assert exited.value.code == 2
    error = capsys.readouterr().err
    assert "Selection exceeds" in error and "Traceback" not in error
