"""Local panel threading/lifetime; actual worker/store/Qt gate is tested separately."""
import hashlib
from threading import Event
from types import SimpleNamespace

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QDialog

from kraken_hub.presentation.qt.local_analysis_maps import LocalAnalysisMapController, _LOCAL_MAP_CALLS
from kraken_manager.presentation.qt.analysis_ui import AnalysisRunsPanel
from tests.test_electrical_graph import png


class Coordinator:
    def __init__(self):
        self.wait = Event()
        self.wait.set()
        self.cancelled = []
        self.data = png([[255, 0], [0, 255]])
        self.store = SimpleNamespace(
            list_runs=lambda: (),
            map_tasks=lambda _id: {"frame": {"state": "ready", "manifest": {"frames": [{"artifacts": [{
                "binding_key": "A", "artifact_version_id": "pinned-version", "sha256": hashlib.sha256(self.data).hexdigest()}]}]}}},
            map_bytes=lambda *_args: self.data,
            get_run=lambda _id: SimpleNamespace(manifest=SimpleNamespace(recipe=SimpleNamespace(
                expression=SimpleNamespace(to_payload=lambda: {"source": "A"})))),
        )

    def request_map(self, *args, **kwargs):
        assert self.wait.wait(5)

    def refresh_maps(self, _id):
        pass

    def cancel(self, identifier):
        assert self.wait.wait(5)
        self.cancelled.append(identifier)


def shown(qtbot):
    widget = QWidget()
    widget.analysis_runs_panel = AnalysisRunsPanel(widget)
    QVBoxLayout(widget).addWidget(widget.analysis_runs_panel)
    qtbot.addWidget(widget)
    widget.show()
    coordinator = Coordinator()
    controller = LocalAnalysisMapController(widget, coordinator, language="English")
    qtbot.waitUntil(lambda: controller._worker is None)
    return widget, coordinator, controller


def test_map_and_cancel_calls_leave_qt_responsive(qtbot):
    widget, coordinator, controller = shown(qtbot)
    coordinator.wait.clear()
    controller.render("run", "frame")
    responsive = []
    QTimer.singleShot(0, lambda: responsive.append(True))
    qtbot.waitUntil(lambda: bool(responsive))
    assert controller._worker is not None
    coordinator.wait.set()
    qtbot.waitUntil(lambda: widget.findChild(QDialog, "localAnalysisMapDialog") is not None, timeout=5000)
    qtbot.waitUntil(lambda: controller._worker is None)
    coordinator.wait.clear()
    controller.command("run", "cancel")
    QTimer.singleShot(0, lambda: responsive.append(True))
    qtbot.waitUntil(lambda: len(responsive) == 2)
    assert controller._worker is not None
    coordinator.wait.set()
    qtbot.waitUntil(lambda: coordinator.cancelled == ["run"])
    qtbot.waitUntil(lambda: controller._worker is None)


def test_parent_destruction_retains_running_local_map_thread(qtbot):
    widget, coordinator, controller = shown(qtbot)
    coordinator.wait.clear()
    controller.render("run", "frame")
    assert controller._worker is not None
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert sip.isdeleted(widget)
    coordinator.wait.set()
    qtbot.waitUntil(lambda: not _LOCAL_MAP_CALLS)
