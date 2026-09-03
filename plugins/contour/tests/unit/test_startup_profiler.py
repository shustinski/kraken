from __future__ import annotations

from unittest.mock import patch

from PyQt6.QtCore import QTimer

from contour.infrastructure.startup_profiler import StartupProfile


def test_startup_profile_reports_phases_and_finishes_once() -> None:
    reports: list[tuple[object, ...]] = []

    with (
        patch("contour.infrastructure.startup_profiler.startup_profiling_enabled", return_value=True),
        patch(
            "contour.infrastructure.startup_profiler.write_profile_report",
            side_effect=lambda *messages: reports.append(messages),
        ),
    ):
        profile = StartupProfile.begin()
        assert profile is not None
        assert profile.measure("application_build", lambda: "window") == "window"
        profile.finish(status="interactive")
        profile.finish(status="failed")

    assert len(reports) == 1
    output = "\n".join(str(message) for message in reports[0])
    assert "[contour startup profiling] status=interactive" in output
    assert "application_build=" in output
    assert "[contour startup profiling stats]" in output


def test_cli_profiles_launch_through_first_event_loop(monkeypatch) -> None:
    from contour import batch_processor, kraken_bridge
    from contour.application import bootstrap, cli
    from contour.infrastructure import startup_profiler

    class FakeApplication:
        def exec(self) -> None:
            return None

    class FakeWindow:
        def show(self) -> None:
            return None

    reports: list[tuple[object, ...]] = []
    monkeypatch.setattr(startup_profiler, "startup_profiling_enabled", lambda: True)
    monkeypatch.setattr(startup_profiler, "write_profile_report", lambda *messages: reports.append(messages))
    monkeypatch.setattr(batch_processor, "configure_batch_runtime", lambda: None)
    monkeypatch.setattr(kraken_bridge, "prepare_contour_launch", lambda args: (None, args))
    monkeypatch.setattr(bootstrap, "build_application", lambda _args: (FakeApplication(), FakeWindow()))
    monkeypatch.setattr(QTimer, "singleShot", lambda _delay, callback: callback())

    cli.main([])

    assert len(reports) == 1
    output = "\n".join(str(message) for message in reports[0])
    assert "[contour startup profiling] status=interactive" in output
    assert "batch_runtime_import=" in output
    assert "batch_runtime_configure=" in output
    assert "launch_bridge_import=" in output
    assert "prepare_launch=" in output
    assert "bootstrap_import=" in output
    assert "application_build=" in output
    assert "window_show=" in output
    assert "first_event_loop=" in output
