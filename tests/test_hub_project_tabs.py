"""Multi-project tab state for the desktop controller."""
from __future__ import annotations
import os
import threading
from pathlib import Path
from uuid import uuid4
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication, QMessageBox

from kraken_hub.application.notifications import ProjectEventWake
from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.presentation.qt import manager_app
from kraken_hub.presentation.qt.manager_app import DesktopController
from kraken_manager.domain.project import GridOrientation
from kraken_manager.presentation.qt import FrameSelection, ProjectListItem, ProjectManagerShell


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def environment(qapp, tmp_path, monkeypatch):
    class FakeRuntime:
        def __init__(self, *_args, **_kwargs) -> None:
            self.process = None
            self.base_url = ""
            self.token = ""
        def ensure_started(self):
            return frozenset()
        def shutdown(self) -> None:
            return None
        def detach(self) -> None:
            return None

    monkeypatch.setattr(manager_app, "LocalAgentRuntime", FakeRuntime)
    service = EmbeddedProjectService(tmp_path)
    session = service.create_initial_account("admin", "Administrator", "")
    shell = ProjectManagerShell()
    controller = DesktopController(shell, service, session, thumbnail_store_uri="memory://", plugin_items=[])
    controller._agent_timer.stop()
    yield controller, shell, service, session
    controller._sync_timer.stop()
    for workspace in list(getattr(controller, "_project_workspaces", {}).values()):
        timer = getattr(workspace, "_analysis_timer", None)
        if timer is not None:
            timer.stop()
    controller.shell.close_guard = None
    shell.close()


def _create_project(service, session, name, root: Path):
    sources = root / f"src-{name}"
    derived = root / f"der-{name}"
    sources.mkdir()
    derived.mkdir()
    return service.create_project(
        principal=session.principal,
        name=name,
        width=4,
        height=3,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key=str(uuid4()),
        layer_template=True,
        source_root=sources,
        derived_root=derived,
    )


def _item(project) -> ProjectListItem:
    return ProjectListItem(str(project.id), project.name, project.width, project.height, "local")

def test_opening_two_projects_keeps_both_tabs_alive(environment, tmp_path):
    controller, shell, service, session = environment
    p1 = _create_project(service, session, "Alpha", tmp_path)
    p2 = _create_project(service, session, "Beta", tmp_path)
    controller.open_project(_item(p1))
    ws1 = controller._workspace
    ws1.matrix_view.set_selection(FrameSelection.single(1, 1))
    controller.open_project(_item(p2))
    assert shell.projectTabs.count() == 2
    assert shell.current_project_id() == str(p2.id)
    assert controller._project_workspaces[str(p1.id)] is ws1
    shell.projectTabs.setCurrentIndex(0)
    assert controller._project_id == str(p1.id)
    assert controller._workspace is ws1
    assert len(list(ws1.matrix_view.selection().coordinates())) == 1  # selection survives the switch


def test_reopening_open_project_reuses_workspace(environment, tmp_path):
    controller, shell, service, session = environment
    p1 = _create_project(service, session, "Alpha", tmp_path)
    controller.open_project(_item(p1))
    ws1 = controller._workspace
    controller.open_project(_item(p1))
    assert controller._workspace is ws1
    assert shell.projectTabs.count() == 1


def test_layer_manager_dialog_closes_on_tab_switch(environment, tmp_path):
    controller, shell, service, session = environment
    p1 = _create_project(service, session, "Alpha", tmp_path)
    p2 = _create_project(service, session, "Beta", tmp_path)
    controller.open_project(_item(p1))
    controller.open_layer_manager()
    assert controller._layer_dialog is not None
    controller.open_project(_item(p2))
    assert controller._layer_dialog is None


def test_layer_manager_open_profile_reports_visible_dialog(environment, tmp_path, monkeypatch):
    from kraken_core.frame_matrix import profiling

    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text("[profiling]\nlayer_manager = true\n", encoding="utf-8")
    profiling.configure(tmp_path)
    controller, shell, service, session = environment
    project = _create_project(service, session, "Profile", tmp_path)
    controller.open_project(_item(project))

    controller.open_layer_manager()
    QApplication.processEvents()

    report = (tmp_path / "profiling.log").read_text(encoding="utf-8")
    assert "[layer_manager]" in report
    assert "pipeline_ready:" in report
    assert "ready=yes" in report


def test_wakes_for_open_inactive_project_are_retained_and_applied_on_activation(environment, tmp_path):
    controller, shell, service, session = environment
    p1 = _create_project(service, session, "Alpha", tmp_path)
    p2 = _create_project(service, session, "Beta", tmp_path)
    controller.open_project(_item(p1))
    controller.open_project(_item(p2))  # p2 active
    ws1 = controller._project_workspaces[str(p1.id)]
    generation_before = ws1.matrix_view.session().generation
    wake = ProjectEventWake(project_id=str(p1.id), event_type="Layer.updated", event_id="e-1")
    controller._on_remote_wake(wake)
    controller._sync_timer.stop()
    controller._drain_live_wakes()
    assert any(w.project_id == str(p1.id) for w in controller._pending_wakes.values())
    assert ws1.matrix_view.session().generation == generation_before
    stale = ProjectEventWake(project_id="ghost", event_type="Layer.updated", event_id="e-2")
    controller._on_remote_wake(stale)
    controller._drain_live_wakes()
    assert not any(w.project_id == "ghost" for w in controller._pending_wakes.values())
    shell.projectTabs.setCurrentIndex(0)  # activate p1
    assert ws1.matrix_view.session().generation == generation_before + 1
    assert not any(w.project_id == str(p1.id) for w in controller._pending_wakes.values())


def test_activated_tab_for_deleted_project_closes_gracefully(environment, tmp_path, monkeypatch):
    controller, shell, service, session = environment
    p1 = _create_project(service, session, "Alpha", tmp_path)
    p2 = _create_project(service, session, "Beta", tmp_path)
    controller.open_project(_item(p1))
    controller.open_project(_item(p2))
    real_get = service.get_project
    monkeypatch.setattr(
        service, "get_project",
        lambda pid: None if str(pid) == str(p1.id) else real_get(pid),
    )
    shell.projectTabs.setCurrentIndex(0)  # p1 is gone -> its tab must close, p2 becomes active
    assert str(p1.id) not in controller._project_workspaces
    assert shell.projectTabs.count() == 1
    assert controller._project_id == str(p2.id)


def test_closing_remote_project_tab_unsubscribes(monkeypatch):
    from types import SimpleNamespace

    controller = object.__new__(DesktopController)
    closed, closed_tabs = [], []

    class ServiceStub:
        is_remote_project = staticmethod(lambda _id: True)

        def unsubscribe_project(self, project_id):
            closed.append(project_id)

    class ShellStub:
        def close_project_tab(self, project_id):
            closed_tabs.append(project_id)

    workspace = SimpleNamespace(
        matrix_view=SimpleNamespace(close=lambda: None),
    )
    controller.service = ServiceStub()
    controller.shell = ShellStub()
    controller._project_workspaces = {"p-r": workspace}
    controller._project_id = "p-r"
    controller._workspace = workspace
    controller._layer_dialog = None

    controller._on_project_tab_close_requested("p-r")

    assert closed == ["p-r"]
    assert closed_tabs == ["p-r"]
    assert controller._project_id is None
    assert controller._workspace is None


def test_closing_project_tab_tolerates_service_without_unsubscribe():
    from types import SimpleNamespace

    controller = object.__new__(DesktopController)

    class ServiceStub:  # local service: no unsubscribe_project at all
        is_remote_project = staticmethod(lambda _id: False)

    class ShellStub:
        def close_project_tab(self, project_id):
            pass

    controller.service = ServiceStub()
    controller.shell = ShellStub()
    controller._project_workspaces = {"p-l": SimpleNamespace(matrix_view=SimpleNamespace(close=lambda: None))}
    controller._project_id = None
    controller._workspace = None
    controller._layer_dialog = None

    controller._on_project_tab_close_requested("p-l")  # must not raise

    assert controller._project_workspaces == {}


def test_close_guard_blocks_with_multiple_open_projects(environment, tmp_path, monkeypatch):
    controller, shell, service, session = environment
    p1 = _create_project(service, session, "Alpha", tmp_path)
    p2 = _create_project(service, session, "Beta", tmp_path)
    controller.open_project(_item(p1))
    controller.open_project(_item(p2))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))

    class Working(QThread):
        def __init__(self, release):
            super().__init__()
            self._release = release

        def run(self):
            while not self._release.is_set():
                self.msleep(20)

    release = threading.Event()
    worker = Working(release)
    worker.start()
    controller._background_workers.add(worker)
    assert controller._allow_close() is False
    release.set()
    worker.wait(5000)
    controller._background_workers.discard(worker)
    assert controller._allow_close() is True
