# ruff: noqa: E402 -- configure Qt and install crash diagnostics before app imports.
import json
import faulthandler
import os
from pathlib import Path
import time
import sys
import traceback
from types import SimpleNamespace

os.environ["QT_QPA_PLATFORM"] = "offscreen"
faulthandler.enable()
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
print("Qt probe: imports", flush=True)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtTest import QTest, QSignalSpy
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import qInstallMessageHandler


def qt_message(kind, context, message):
    print(f"Qt message {kind}: {message}", flush=True)


uncaught_errors = []


def uncaught(kind, value, tb):
    uncaught_errors.append(str(value))
    traceback.print_exception(kind, value, tb)
    sys.stderr.flush()


qInstallMessageHandler(qt_message)
sys.excepthook = uncaught
from kraken_core.settings import configure_settings_directory
from kraken_core.plugins import PluginInventoryItem, load_plugin_metadata_from_directory
from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.composition.catalog import DualCatalogService
from kraken_hub.presentation.qt.manager_app import DesktopController
from kraken_manager.presentation.qt import ProjectManagerShell
from live_server_context import RUN, connect

configure_settings_directory(RUN / "qt-settings")
app = QApplication([])
app.setQuitOnLastWindowClosed(False)
from PyQt6.QtGui import QFontDatabase, QFont
QFontDatabase.addApplicationFont("C:/Windows/Fonts/arial.ttf")
app.setFont(QFont("Arial", 10))
remote = connect()
print("Qt probe: connected", flush=True)
project = remote.list_projects()[0]
before = [str(e.event_id) for e in remote.history(project.id)]
local = EmbeddedProjectService(RUN / "hub-local-cache")
dual = DualCatalogService(local, remote)
plugins = [PluginInventoryItem(load_plugin_metadata_from_directory(Path(__file__).resolve().parents[3] / "plugins" / name))
           for name in ("contour", "neuralimage", "karakal")]
notices = []
pump = QTimer()
pump.setInterval(10)


def dismiss_unexpected():
    box = QApplication.activeModalWidget()
    if isinstance(box, QMessageBox):
        notices.append({"title": box.windowTitle(), "text": box.text()})
        box.accept()


pump.timeout.connect(dismiss_unexpected)
pump.start()


def wait(predicate, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QTest.qWait(20)
        if predicate():
            return
    raise AssertionError("Qt state did not become ready")


receipt = {"project_id": str(project.id), "status": "running", "passes": []}
shell = None
try:
    for attempt in range(2):
        print(f"Qt probe: shell {attempt}", flush=True)
        shell = ProjectManagerShell()
        print("Qt probe: shell built", flush=True)
        shell.resize(1440, 1000)
        controller = DesktopController(shell, dual, SimpleNamespace(principal=remote.auth.principal),
                                       thumbnail_store_uri="memory://", plugin_items=plugins)
        print("Qt probe: controller built", flush=True)
        controller._agent_timer.stop()
        shell.show()
        app.processEvents()
        print("Qt probe: shell events drained", flush=True)
        capabilities = controller.agent_runtime.ensure_started()
        print("Qt probe: agent ready", flush=True)
        receipt["agent"] = {"url": controller.agent_runtime.base_url,
                            "capabilities": sorted(capabilities),
                            "health": controller.agent_runtime._request("GET", "/api/v1/health")["status"]}
        view = controller.catalog_page.project_list
        index = controller.catalog_page.project_model.index(0, 0)
        assert index.isValid() and view.isVisible()
        view.setCurrentIndex(index)
        view.scrollTo(index)
        app.processEvents()
        activated = QSignalSpy(controller.catalog_page.projectActivated)
        print("Qt probe: index", index.data(), view.visualRect(index), view.viewport().rect(), flush=True)
        shell.grab().save(str(RUN / "catalog-before-click.png"))
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=view.visualRect(index).center())
        print("Qt probe selected", view.currentIndex().data(), flush=True)
        index = controller.catalog_page.project_model.index(0, 0)
        print("Qt probe point", view.visualRect(index), flush=True)
        QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, pos=view.visualRect(index).center())
        print("Qt probe: double click delivered", len(activated), "notices", notices, flush=True)
        wait(lambda: controller._workspace is not None)
        workspace = controller._workspace
        wait(lambda: workspace.image_representation_combo.count() > 0 and workspace.vector_representation_combo.count() > 0)
        wait(lambda: not controller._background_workers)
        QTest.qWait(1500)
        assert controller._project_id == str(project.id)
        assert dual.is_remote_project(project.id)
        assert workspace.layer_tabs.count() == 1
        screenshot = RUN / f"project-view-{attempt + 1}.png"
        assert shell.grab().save(str(screenshot))
        receipt["passes"].append({"opened_by": "QTest.mouseDClick catalog item", "layers": workspace.layer_tabs.count(),
                                 "images": workspace.image_representation_combo.count(),
                                 "vectors": workspace.vector_representation_combo.count(), "screenshot": str(screenshot)})
        workspace._analysis_timer.stop()
        wait(lambda: not controller._background_workers)
        shell.close()
        app.processEvents()
        shell = None
    after = [str(e.event_id) for e in remote.history(project.id)]
    receipt["project_history_unchanged"] = before == after
    receipt["revision_before"] = project.revision
    receipt["revision_after"] = remote.get_project(project.id).revision
    receipt["notices"] = notices
    receipt["uncaught_errors"] = uncaught_errors
    assert not uncaught_errors, uncaught_errors
    assert not notices, notices
    assert receipt["project_history_unchanged"]
    receipt["status"] = "passed"
finally:
    pump.stop()
    if shell is not None:
        shell.close()
        app.processEvents()
    dual.stop_sync()
    (RUN / "qt-project-view.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
print(json.dumps(receipt, ensure_ascii=False))
