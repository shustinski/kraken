"""Settle background QThreads started by UI tests before the process continues."""

from __future__ import annotations

import os

# Force pytest-qt to bind to the same Qt used by Karakal (PyQt6).
os.environ.setdefault("PYTEST_QT_API", "pyqt6")

import pytest


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
    for _attempt in range(50):
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
