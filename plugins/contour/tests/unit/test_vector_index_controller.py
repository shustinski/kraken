from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from contour.application.services import VectorIndexController


def _app() -> QApplication:
    instance = QApplication.instance()
    return instance if instance is not None else QApplication([])


def _wait_for(condition, *, timeout_ms: int = 1000) -> bool:
    app = _app()
    elapsed = 0
    while elapsed < timeout_ms:
        app.processEvents()
        if condition():
            return True
        QTest.qWait(10)
        elapsed += 10
    return condition()


def test_vector_index_controller_ignores_stale_busy_result_and_runs_pending_directory() -> None:
    _app()
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        Path(first, "first.cif").write_text("placeholder", encoding="utf-8")
        Path(second, "second.cv").write_text("placeholder", encoding="utf-8")
        controller = VectorIndexController()
        finished = []
        controller.finished.connect(finished.append)

        controller.start(first)
        controller.start(second)

        assert _wait_for(lambda: not controller.busy and len(finished) == 1)
        assert finished[0].directory == str(Path(second))
        assert set(finished[0].indexed_paths) == {"second"}
