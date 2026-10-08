"""Main run button (run / analyze / recompute / stop), Shift for frames only, pause of a task."""

from __future__ import annotations

import threading
import time

import numpy as np
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from karakal.core.image_io import _grayscale_array_to_qimage
from karakal.core.workers import WorkerBase


def test_paused_worker_waits_at_its_checkpoint_and_cancel_releases_it() -> None:
    worker = WorkerBase()
    worker.request_pause()
    assert worker.paused
    passed: list[float] = []

    def loop() -> None:
        started = time.perf_counter()
        worker._is_cancelled()
        passed.append(time.perf_counter() - started)

    thread = threading.Thread(target=loop)
    thread.start()
    time.sleep(0.4)
    assert not passed
    worker.request_resume()
    thread.join(2.0)
    assert passed and passed[0] >= 0.35
    worker.request_pause()
    thread = threading.Thread(target=loop)
    thread.start()
    worker.request_cancel()
    thread.join(2.0)
    assert not thread.is_alive() and worker._is_cancelled()


def _widget(tmp_path, qtbot):
    from karakal.app.main_window import KarakalWidget

    widget = KarakalWidget(settings=QSettings(str(tmp_path / "k.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    return widget


def _add_folders(tmp_path, presenter, names) -> None:
    for name in names:
        folder = tmp_path / name
        folder.mkdir(parents=True, exist_ok=True)
        assert _grayscale_array_to_qimage(np.full((8, 8), 255, dtype=np.uint8)).save(str(folder / "F_0001.png"))
        presenter._append_folder_item(folder, checked=True)
    presenter._refresh_folder_rows()
    presenter._sync_action_buttons()


def test_setup_panel_has_no_banner_and_no_run_button(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)
    panel = widget.analysis_setup_panel
    assert not hasattr(panel, "run_button")
    assert not hasattr(panel, "preflight_label")
    assert not hasattr(panel, "workflow_label")
    assert widget.folders_group.title() == ""
    titles = [button.text() for button in panel._profile_buttons.values()]
    assert widget._t("profile.grid_defects.title") in titles
    assert widget._t("profile.grid_defects.title") in ("Поиск дефектов матрицы", "Matrix defect search")


def test_not_ready_run_explains_why(tmp_path, qtbot, monkeypatch) -> None:
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter
    presenter._sync_action_buttons()
    assert widget.btn_run.property("runState") == "blocked"
    assert widget.btn_run.isEnabled()
    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "information", lambda _parent, _title, text: shown.append(text))
    started: list[bool] = []
    monkeypatch.setattr(presenter, "_on_primary_run_requested", lambda: started.append(True))
    widget.btn_run.click()
    assert shown and not started
    assert widget.folders_empty_hint.isVisibleTo(widget)


def test_ready_run_starts_the_full_run_and_shift_only_gathers_frames(tmp_path, qtbot, monkeypatch) -> None:
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter
    _add_folders(tmp_path, presenter, ("a/result", "b/result"))
    assert widget.btn_run.property("runState") == "go"
    assert widget.btn_run.text() == widget._t("run.run")
    calls: list[str] = []
    monkeypatch.setattr(presenter, "_on_primary_run_requested", lambda: calls.append("run"))
    monkeypatch.setattr(presenter, "_on_build_requested", lambda: calls.append("build"))
    widget.btn_run.click()
    monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: Qt.KeyboardModifier.ShiftModifier))
    widget.btn_run.click()
    assert calls == ["run", "build"]
    assert not widget.folders_empty_hint.isVisibleTo(widget)


def test_running_task_shows_stop_with_percent_and_pause(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter
    _add_folders(tmp_path, presenter, ("a/result", "b/result"))
    worker = WorkerBase()
    presenter._worker = worker
    presenter._worker_thread = object()
    try:
        presenter._show_progress_bar(visible=True, current=45, total=100, key="F_0045.png")
        assert widget.btn_run.property("runState") == "stop"
        assert "45" in widget.btn_run.text()
        assert widget.btn_pause.isVisibleTo(widget)
        widget.btn_pause.click()
        assert worker.paused
        assert widget.btn_pause.text() == widget._t("run.resume")
        widget.btn_pause.click()
        assert not worker.paused
        widget.btn_run.click()
        assert worker._cancel_requested.is_set()
    finally:
        presenter._worker = None
        presenter._worker_thread = None
        presenter._show_progress_bar(visible=False)
        presenter._sync_action_buttons()
    assert not widget.btn_pause.isVisibleTo(widget)
