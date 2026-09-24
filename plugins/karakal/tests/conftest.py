"""Settle background QThreads started by UI tests before the process continues."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _skip_update_checks(monkeypatch):
    from karakal.app.main_window import QtUpdateController

    monkeypatch.setattr(QtUpdateController, "check_for_updates", lambda self, manual=False: None)


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
