"""Settle background QThreads started by UI tests before the process continues."""

from __future__ import annotations

import os

# Force pytest-qt to bind to the same Qt used by Karakal (PyQt6).
os.environ.setdefault("PYTEST_QT_API", "pyqt6")
# Windows and message boxes of the tests never reach the desktop: they used to pop up over the
# user's work (a modal box then waits for a click), and parallel runs showing real windows in
# several processes now and then crashed Qt. The offscreen platform needs the font folder.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Tests drive the workspace directly; the home page is tested on its own.
os.environ.setdefault("KARAKAL_HOME_PAGE", "0")
# Dirt found after a cell check runs in threads, not in a process pool per test.
os.environ.setdefault("KARAKAL_DIRT_PROCESSES", "0")
if os.name == "nt" and os.path.isdir(r"C:\Windows\Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import pytest


@pytest.fixture(autouse=True)
def _own_bank_cache(tmp_path_factory, monkeypatch):
    """Banks of normal cells built by the tests are kept in a folder of the test run, not in the
    user's cache."""

    import karakal.core.grid_anomaly as grid_anomaly

    monkeypatch.setattr(grid_anomaly, "RUN_PROFILE_CACHE_DIR", tmp_path_factory.mktemp("normal_banks"), raising=False)


@pytest.fixture(autouse=True)
def _own_dirt_cache_folder(tmp_path_factory, monkeypatch):
    """Dirt found by the tests (after every cell check) is cached in a folder of the test run."""

    import karakal.core.dirt_batch as dirt_batch

    folder = tmp_path_factory.mktemp("dirt_cache")
    monkeypatch.setattr(dirt_batch, "default_cache_dir", lambda: folder)


@pytest.fixture(autouse=True)
def _own_projects_folder(tmp_path_factory, monkeypatch):
    """Projects saved by the tests (every finished run saves one) go to a folder of the test run."""

    monkeypatch.setenv("KARAKAL_PROJECTS_DIR", str(tmp_path_factory.mktemp("projects")))


@pytest.fixture(autouse=True)
def _own_data_folder(tmp_path_factory, monkeypatch):
    """Profiles and exports of the tests go to a data folder of the test run, not to Documents."""

    import karakal.core.data_folder as data_folder

    monkeypatch.setenv("KARAKAL_DATA_DIR", str(tmp_path_factory.mktemp("karakal_data")))
    monkeypatch.setattr(data_folder, "_root", None)


@pytest.fixture(autouse=True)
def _skip_update_checks(monkeypatch):
    """No background look into the real update folder; update tests call check() themselves."""

    from karakal.app.update_flow import KarakalUpdateManager

    monkeypatch.setattr(KarakalUpdateManager, "start", lambda self, first_check_delay_ms=1500: None)


@pytest.fixture(autouse=True)
def _settle_background_qthreads(request):
    yield
    if "qtbot" not in request.fixturenames:
        return
    from PyQt6.QtWidgets import QApplication

    from karakal.app.presenter import _PRESENTER_ORPHAN_THREADS
    from karakal.ui.details_dialog import _ORPHAN_THREADS

    app = QApplication.instance()
    # Up to 30 s: under a parallel run a frame window's thread analyses its frame several times slower
    # than alone (5 s was too short and failed a test now and then); a thread that never ends still fails.
    for _attempt in range(300):
        running = [
            thread
            for thread, _worker in (*_ORPHAN_THREADS, *_PRESENTER_ORPHAN_THREADS)
            if thread.isRunning()
        ]
        if not running:
            if app is not None:
                app.processEvents()
            return
        if app is not None:
            app.processEvents()
        for thread in running:
            thread.wait(100)
    leftovers = [
        thread
        for thread, _worker in (*_ORPHAN_THREADS, *_PRESENTER_ORPHAN_THREADS)
        if thread.isRunning()
    ]
    assert leftovers == [], f"background QThread still running after {request.node.name}"
