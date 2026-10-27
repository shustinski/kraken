"""Qt transport/control behavior; actual Karakal execution has a separate gate."""
import copy
import hashlib
from threading import Event

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QDialog, QLabel

from kraken_hub.remote_analysis import RemoteAnalysisController, _ACTIVE_CALLS
from kraken_manager.presentation.qt.analysis_ui import AnalysisRunsPanel


class Service:
    def __init__(self):
        self.runs = {}
        self.commands = []
        self.created = {}
        self.wait = Event()
        self.wait.set()
        self.lose_reply = False

    def server_analysis_runs(self, project_id):
        assert self.wait.wait(5)
        return tuple(copy.deepcopy(tuple(self.runs.values())))

    def server_analysis_frames(self, project_id, run_id, metric):
        return ({"frame_id": "frame", "x": 1, "y": 1, "status": "ready", "message": "transport fixture",
                 "raw_value": 0.5, "goodness": 0.5, "percentile": None},)

    def create_server_analysis(self, project_id, payload, *, idempotency_key):
        assert self.wait.wait(5)
        self.commands.append(("create", idempotency_key, copy.deepcopy(payload)))
        if idempotency_key not in self.created:
            identifier = str(len(self.runs) + 1)
            self.created[idempotency_key] = identifier
            self.runs[identifier] = {"run_id": identifier, "state": "running", "revision": 1, "completed_frames": 0,
                "failed_frames": 0, "total_frames": 1, "manifest": {
                    "source_bindings": [{"binding_key": "A", "source_id": "image-representation", "display_name": "Mask A"}],
                    "recipe": payload["recipe"]}}
        if self.lose_reply:
            self.lose_reply = False
            raise OSError("reply lost after commit")
        return copy.deepcopy(self.runs[self.created[idempotency_key]])

    def command_server_analysis(self, project_id, run_id, operation, *, expected_revision, idempotency_key):
        assert self.runs[run_id]["revision"] == expected_revision
        self.commands.append((operation, idempotency_key, expected_revision))
        self.runs[run_id]["state"] = "cancelled" if operation == "cancel" else "running"
        self.runs[run_id]["revision"] += 1
        return copy.deepcopy(self.runs[run_id])

    def request_server_analysis_map(self, project_id, run_id, frame_id, *, expected_revision, idempotency_key):
        assert self.wait.wait(5)
        assert self.runs[run_id]["revision"] == expected_revision
        from tests.test_electrical_graph import png
        data = png([[0, 255], [255, 0]])
        self.runs[run_id].setdefault("maps", {})[frame_id] = {
            "state": "ready", "blob": {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}}
        self.runs[run_id]["revision"] += 1
        return copy.deepcopy(self.runs[run_id])

    def server_analysis_map_bytes(self, project_id, run_id, frame_id):
        assert self.runs[run_id]["maps"][frame_id]["state"] == "ready"
        from tests.test_electrical_graph import png
        return png([[0, 255], [255, 0]])


def shown(qtbot, service):
    widget = QWidget()
    widget.analysis_runs_panel = AnalysisRunsPanel(widget)
    QVBoxLayout(widget).addWidget(widget.analysis_runs_panel)
    qtbot.addWidget(widget)
    widget.show()
    controller = RemoteAnalysisController(widget, service, "project", language="English")
    qtbot.waitUntil(lambda: controller._worker is None)
    return widget, controller


CONFIGURATION = {"bindings": {"A": "image-representation"}, "recipe": {"expression": {"source": "A"}}}


def test_remote_panel_commands_and_csv_use_background_calls(qtbot, tmp_path, monkeypatch):
    service = Service()
    widget, controller = shown(qtbot, service)
    service.wait.clear()
    controller.start(CONFIGURATION, "layer", ((1, 1),))
    responsive = []
    QTimer.singleShot(0, lambda: responsive.append(True))
    qtbot.waitUntil(lambda: bool(responsive))
    assert controller._worker is not None
    service.wait.set()
    panel = widget.analysis_runs_panel
    qtbot.waitUntil(lambda: panel.run_table.rowCount() == 1 and panel.result_table.rowCount() == 1, timeout=5000)
    assert service.commands[0][2]["coordinates"] == [[1, 1]]
    assert panel.selected_run_id() == "1" and panel.run_table.item(0, 2).text() == "A=Mask A"
    qtbot.waitUntil(lambda: controller._worker is None)
    controller.command("1", "cancel")
    qtbot.waitUntil(lambda: panel.run_table.item(0, 0).text() == "cancelled", timeout=5000)
    assert service.commands[-1][0] == "cancel" and service.commands[-1][2] == 1
    qtbot.waitUntil(lambda: controller._worker is None)
    target = tmp_path / "analysis.csv"
    monkeypatch.setattr("kraken_hub.remote_analysis.QFileDialog.getSaveFileName", lambda *a: (str(target), "CSV"))
    controller.export("1")
    qtbot.waitUntil(target.exists)
    assert "transport fixture" in target.read_text(encoding="utf-8-sig")
    qtbot.waitUntil(lambda: controller._worker is None)


def test_lost_create_reply_reuses_original_idempotency_key(qtbot):
    service = Service()
    widget, controller = shown(qtbot, service)
    service.lose_reply = True
    controller.start(CONFIGURATION, "layer", ((1, 1),))
    qtbot.waitUntil(lambda: controller.retry_request.isEnabled())
    qtbot.waitUntil(lambda: controller._worker is None)
    controller.retry_request.click()
    qtbot.waitUntil(lambda: widget.analysis_runs_panel.run_table.rowCount() == 1, timeout=5000)
    qtbot.waitUntil(lambda: controller._worker is None)
    assert len(service.runs) == 1 and service.commands[0][1] == service.commands[1][1]
    assert not controller.retry_request.isEnabled()


def test_parent_destruction_during_remote_call_does_not_destroy_running_qthread(qtbot):
    service = Service()
    widget, controller = shown(qtbot, service)
    service.wait.clear()
    controller.refresh()
    assert controller._worker is not None
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert sip.isdeleted(widget)
    service.wait.set()
    qtbot.waitUntil(lambda: not _ACTIVE_CALLS, timeout=5000)


def test_remote_map_button_requests_and_shows_png_in_nonmodal_window(qtbot):
    service = Service()
    widget, controller = shown(qtbot, service)
    controller.start(CONFIGURATION, "layer", ((1, 1),))
    panel = widget.analysis_runs_panel
    qtbot.waitUntil(lambda: panel.result_table.rowCount() == 1)
    qtbot.waitUntil(lambda: controller._worker is None)
    panel.result_table.selectRow(0)
    panel.map_button.click()
    qtbot.waitUntil(lambda: widget.findChild(QDialog, "remoteAnalysisMapDialog") is not None, timeout=5000)
    dialog = widget.findChild(QDialog, "remoteAnalysisMapDialog")
    image = dialog.findChild(QLabel, "analysisMapImage")
    assert dialog.isVisible() and not dialog.isModal()
    assert image.pixmap().width() == 2 and image.pixmap().height() == 2
    assert image.pixmap().toImage().pixelColor(1, 0).red() == 255
    qtbot.waitUntil(lambda: controller._worker is None)
    dialog.close()


def test_remote_map_checksum_failure_does_not_show_a_wrong_image(qtbot):
    service = Service()
    widget, controller = shown(qtbot, service)
    controller.start(CONFIGURATION, "layer", ((1, 1),))
    panel = widget.analysis_runs_panel
    qtbot.waitUntil(lambda: panel.result_table.rowCount() == 1)
    qtbot.waitUntil(lambda: controller._worker is None)
    service.server_analysis_map_bytes = lambda *_args: b"tampered transfer"
    panel.result_table.selectRow(0)
    panel.map_button.click()
    qtbot.waitUntil(lambda: "checksum" in controller.status.text(), timeout=5000)
    assert widget.findChild(QDialog, "remoteAnalysisMapDialog") is None
    qtbot.waitUntil(lambda: controller._worker is None)
