"""Named planning and immutable evidence selection through real services and Qt."""
from threading import Event
from types import SimpleNamespace

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QDialogButtonBox, QLabel

from kraken_hub.planning_evidence_picker import PlanningEvidencePicker
from kraken_hub.work_planning_dialog import WorkPlanningDialog
from kraken_manager.domain.planning import WorkScope
from tests.test_work_planning_dialog import click, wait_done, check_layer, move
from tests.test_work_orders import _service, _project_with_frames

pytest_plugins = ["tests.test_admin_project_workflows"]


def test_named_performers_are_read_from_the_project_backend(qtbot):
    from tests.test_work_planning_dialog import PlanningAdapter

    class ProjectCatalogue(PlanningAdapter):
        def list_performers(self, **kwargs):
            raise AssertionError("The merged local/remote catalogue must not supply project assignments")

        def list_project_performers(self, project_id, **kwargs):
            assert project_id == "selected-project"
            return self.performers

    service = ProjectCatalogue()
    dialog = WorkPlanningDialog(service, SimpleNamespace(id="owner"), "selected-project")
    qtbot.addWidget(dialog)
    dialog.show()
    wait_done(qtbot, dialog)
    assert dialog.performer.count() == 2
    assert dialog.performer.itemData(1) == service.performers[0].id
    dialog.close()


def test_picker_frame_scope_refuses_foreign_frame_and_preserves_checks_after_failure(qtbot, tmp_path, monkeypatch):
    local = _service(tmp_path)
    _, _, project, layer, _, _ = _project_with_frames(local, tmp_path / "source", with_vectors=False)
    scope = WorkScope(layer_ids=(layer.id,), frame_ids=(project.frame_id_at(1, 1),))
    picker = PlanningEvidencePicker(local, project.id, project, [layer], scope, {}, language="English")
    qtbot.addWidget(picker)
    picker.show()
    assert picker.kind.count() == 1
    picker.x.setValue(2)
    picker.load_button.click()
    assert "outside" in picker.status.text() and picker._worker is None
    picker.x.setValue(1)
    picker.load_button.click()
    qtbot.waitUntil(lambda: picker._worker is None)
    assert picker.versions.count() == 1
    row = picker.versions.item(0)
    row.setCheckState(Qt.CheckState.Checked)
    selected = picker.selected_versions
    monkeypatch.setattr(local, "quality_frame_versions", lambda *a: (_ for _ in ()).throw(ConnectionError("Lost evidence response")))
    picker.load_button.click()
    qtbot.waitUntil(lambda: picker._worker is None)
    assert "Lost evidence response" in picker.status.text()
    assert picker.selected_versions == selected and picker.chosen.count() == 1
    picker.chosen.item(0).setSelected(True)
    picker.remove_button.click()
    assert not picker.selected_versions and picker.versions.item(0).checkState() == Qt.CheckState.Unchecked
    picker.close()


def test_picker_loading_participates_in_parent_and_hub_close_guard(qtbot, tmp_path):
    from tests.test_work_planning_dialog import PlanningAdapter
    local = PlanningAdapter()
    workers, release = set(), Event()
    dialog = WorkPlanningDialog(local, SimpleNamespace(id="owner"), "project", background_workers=workers)
    qtbot.addWidget(dialog)
    wait_done(qtbot, dialog)

    def read(*_args):
        assert release.wait(timeout=5)
        return local.versions
    local.quality_frame_versions = read
    picker = PlanningEvidencePicker(local, "project", SimpleNamespace(width=2, height=2), local.layers,
        WorkScope(), {}, dialog, background_workers=workers)
    dialog._evidence_picker = picker
    picker.show()
    try:
        picker.load_button.click()
        qtbot.waitUntil(lambda: any(w.isRunning() for w in workers))
        assert not picker.close() and not dialog.close()
    finally:
        release.set()
        qtbot.waitUntil(lambda: picker._worker is None)
        assert not workers
        picker.close()
        dialog._evidence_picker = None
        dialog.close()


def test_postgres_http_named_work_and_attachment_acceptance_reopen(qtbot, database, tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from PyQt6.QtGui import QFont, QFontDatabase
    from PyQt6.QtWidgets import QApplication
    import socket
    import threading
    import time
    import uvicorn
    from kraken_hub.remote_client import RemoteAuth, RemoteServerProjectService
    from kraken_server.app import SessionPrincipal, create_app
    from tests.test_quality_postgres import setup_quality

    # The isolated pytest APPDATA profile has no Windows font cache. Load the
    # same installed font used by the deployment probe for reviewable PNGs.
    if os.name == "nt":
        font_id = QFontDatabase.addApplicationFont(str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"))
        assert font_id >= 0
        QApplication.instance().setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 9))
    service, owner, pid, performer, _source, corrected, arguments = setup_quality(database)
    lid, = arguments["scope"]["layer_ids"]
    app = create_app(services=service, session_resolver=lambda token: SessionPrincipal(str(owner.id), "local", token) if token == "owner" else None)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    dialog = reopened = None
    try:
        deadline = time.monotonic()+5
        while not server.started and time.monotonic() < deadline:
            time.sleep(.01)
        assert server.started
        remote = RemoteServerProjectService(f"http://127.0.0.1:{sock.getsockname()[1]}",
            auth=RemoteAuth(access_token="owner", principal=owner), data_dir=tmp_path / "remote")
        dialog = WorkPlanningDialog(remote, owner, pid, language="English")
        qtbot.addWidget(dialog)
        dialog.show()
        wait_done(qtbot, dialog)
        assert not any("UUID" in label.text() for label in dialog.findChildren(QLabel))
        dialog.plan_title.setText("Named acceptance")
        click(qtbot, dialog, dialog.create_button)
        assert dialog.plan is not None, dialog.status.text()
        dialog.item_title.setText("Review layer result")
        dialog.performer.setCurrentIndex(dialog.performer.findText(performer.name))
        check_layer(dialog, lid)
        dialog.criterion_description.setText("Reviewed immutable attachment")
        click(qtbot, dialog, dialog.add_button)
        assert dialog.items.item(0, 2).text() == performer.name
        assert dialog.items.item(0, 5).text() == "M1"
        click(qtbot, dialog, dialog.activate_button)
        move(qtbot, dialog, "running")
        dialog.percent.setValue(100)
        click(qtbot, dialog, dialog.progress_button)
        move(qtbot, dialog, "in_review")
        original_exec = PlanningEvidencePicker.exec

        def pick_attachment(picker):
            def fill():
                try:
                    assert picker.isVisible()
                    picker.kind.setCurrentIndex(picker.kind.findData("layer_attachment"))
                    picker.load_button.click()
                    qtbot.waitUntil(lambda: picker._worker is None, timeout=10000)
                    picker.series.setCurrentIndex(picker.series.findText("Measurement"))
                    qtbot.waitUntil(lambda: picker._worker is None, timeout=10000)
                    row = picker.versions.item(0)
                    assert row.data(Qt.ItemDataRole.UserRole) == corrected["id"]
                    assert "measurement.txt" in row.text()
                    row.setCheckState(Qt.CheckState.Checked)
                    picker.buttons.button(QDialogButtonBox.StandardButton.Ok).click()
                finally:
                    if picker.isVisible():
                        picker.reject()
            QTimer.singleShot(0, fill)
            return original_exec(picker)

        with monkeypatch.context() as patch:
            patch.setattr(PlanningEvidencePicker, "exec", pick_attachment)
            click(qtbot, dialog, dialog.evidence_button)
        assert list(dialog._evidence_versions) == [corrected["id"]]
        click(qtbot, dialog, dialog.satisfy_button)
        assert dialog.plan.items[0].criteria[0].evidence_version_ids == (corrected["id"],), dialog.status.text()
        move(qtbot, dialog, "accepted")
        click(qtbot, dialog, dialog.complete_button)
        assert dialog.plan.state == "completed"
        dialog.grab().save(str(tmp_path / "named-planning.png"))
        saved = dialog.plan
        dialog.close()
        reopened = WorkPlanningDialog(remote, owner, pid, language="Русский")
        qtbot.addWidget(reopened)
        reopened.show()
        wait_done(qtbot, reopened)
        assert reopened.plan == saved
        assert list(reopened._evidence_versions) == [corrected["id"]]
        assert reopened.items.item(0, 2).text() == performer.name
        assert "UUID" not in reopened.evidence.text()
        picker = PlanningEvidencePicker(remote, pid, reopened._project, reopened._available_layers,
            saved.items[0].scope, reopened._evidence_versions, language="English")
        qtbot.addWidget(picker)
        picker.show()
        qtbot.waitUntil(lambda: picker._worker is None, timeout=10000)
        assert picker.chosen.count() == 1 and "measurement.txt" in picker.chosen.item(0).text()
        assert corrected["sha256"][:12] in picker.chosen.item(0).text()
        picker.grab().save(str(tmp_path / "recorded-evidence.png"))
        picker.reject()
    finally:
        for window in (reopened, dialog):
            if window is not None:
                wait_done(qtbot, window)
                window.close()
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
