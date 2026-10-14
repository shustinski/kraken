"""test_server_workspace_inputs: test desktop handoff reports."""
from __future__ import annotations
from types import SimpleNamespace
import pytest
from kraken_hub.manager_app import DesktopController

@pytest.mark.parametrize("fails", [False, True])
def test_desktop_handoff_reports_result_and_refreshes_on_qt_thread(qtbot, fails):
    from threading import get_ident
    from PyQt6.QtWidgets import QMainWindow

    controller = object.__new__(DesktopController)
    controller.shell = QMainWindow()
    qtbot.addWidget(controller.shell)
    controller._project_id = "project"
    controller._workspace = SimpleNamespace(_selected_layer_id="layer")
    controller.my_work_refresh = None
    controller._background_workers = set()
    calls = []
    refreshed = []
    errors = []
    qt_thread = get_ident()

    def execute(project_id, job_id, *, registry):
        calls.append((project_id, job_id, registry, get_ident()))
        if fails:
            raise RuntimeError("Contour could not publish the result")
        return {"imported": {"job": {"state": "succeeded"}}}

    controller.service = SimpleNamespace(run_interactive_job=execute)
    controller._load_representations = lambda *_args: refreshed.append(get_ident())
    controller._refresh_layer_pipeline = lambda *_args: refreshed.append(get_ident())
    controller._error = errors.append
    controller._run_remote_interactive_job(
        SimpleNamespace(project_id="project", layer_id="layer", id="job"), registry="registry"
    )
    qtbot.waitUntil(lambda: not controller._background_workers, timeout=3000)
    assert len(calls) == 1
    assert calls[0][:3] == ("project", "job", "registry")
    assert calls[0][3] != qt_thread
    assert refreshed == ([] if fails else [qt_thread, qt_thread])
    assert bool(errors) is fails
