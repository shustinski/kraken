from threading import Event

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QMessageBox

from csliser.application.use_cases import ExecuteTransferPlan
from csliser.domain.models import FileOperation, ProcessingConfig, SelectionMode, SourceFolder
from csliser.domain.planner import build_operation_plan
from csliser.infrastructure.settings_store import WindowSettingsStore
from csliser.presentation.qt.window import CSliserWindow
from csliser.presentation.qt.worker import TransferWorker


def setup(qtbot, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "chip_000001.jpg").write_bytes(b"real transfer bytes")
    plan = build_operation_plan(ProcessingConfig((SourceFolder(source, (".jpg",)),), "1", selection_mode=SelectionMode.FULL_RANGE,
        operation=FileOperation.COPY, destination=tmp_path / "out"))
    window = CSliserWindow(settings_store=WindowSettingsStore(lambda: QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)))
    qtbot.addWidget(window)
    window.show()
    return window, plan


def test_real_transfer_finishes_thread_before_releasing_ui(qtbot, tmp_path, monkeypatch):
    window, plan = setup(qtbot, tmp_path)
    notices = []
    monkeypatch.setattr(QMessageBox, "information", lambda *_args: notices.append(window._worker_thread))
    window._start_worker(plan)
    qtbot.waitUntil(lambda: window._worker_thread is None, timeout=5000)
    assert notices == [None] and window._last_result.ok and not window._busy
    assert plan.operations[0].destination.read_bytes() == b"real transfer bytes"
    window.close()


def test_close_requests_cancel_and_waits_for_thread_before_window_deletion(qtbot, tmp_path, monkeypatch):
    from csliser.presentation.qt import window as module
    window, plan = setup(qtbot, tmp_path)
    entered = Event()
    class BlockingExecutor:
        def execute(self, plan, *, progress, cancelled):
            entered.set()
            while not cancelled():
                Event().wait(.01)
            return ExecuteTransferPlan().execute(plan, progress=progress, cancelled=cancelled)
    monkeypatch.setattr(module, "TransferWorker", lambda p: TransferWorker(p, executor=BlockingExecutor()))
    window._start_worker(plan)
    qtbot.waitUntil(entered.is_set, timeout=5000)
    window.close()
    assert window.isVisible() and window._worker_thread is not None
    qtbot.waitUntil(lambda: window._worker_thread is None and not window.isVisible(), timeout=5000)
    assert window._last_result.cancelled and not plan.operations[0].destination.exists()


def test_unexpected_backend_error_reports_unknown_outcome_and_does_not_leave_ui_busy(qtbot, tmp_path, monkeypatch):
    from csliser.presentation.qt import window as module
    window, plan = setup(qtbot, tmp_path)
    class FailingExecutor:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("controlled unexpected transfer backend failure")
    notices = []
    monkeypatch.setattr(module, "TransferWorker", lambda p: TransferWorker(p, executor=FailingExecutor()))
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: notices.append(True))
    window._start_worker(plan)
    qtbot.waitUntil(lambda: window._worker_thread is None, timeout=5000)
    assert notices and not window._busy and window._last_result.outcome_unknown
    assert not window._last_result.ok and window._last_result.errors
    window.close()
