"""Opt-in acceptance tests against an isolated, running Kraken Server.

Set KRAKEN_UI_TEST_CONFIG to a private JSON file containing server_url,
username, password and source_directory. The source must contain 100 numbered
images. These tests create their own project and never delete the source.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
)

from kraken_core.plugins import build_plugin_inventory, scan_plugin_directory
from kraken_hub.composition import DesktopSession, EmbeddedProjectService
from kraken_hub.dual_catalog import DualCatalogService
from kraken_hub.manager_app import DesktopController, _my_work_panel, _statistics_panel
from kraken_hub.remote_client import RemoteServerProjectService
from kraken_manager.presentation.qt import ProjectManagerShell
from kraken_manager.presentation.qt.layer_creation import LayerCreationDialog


def click(widget, text):
    button = next(button for button in widget.findChildren(QPushButton) if button.text() == text)
    assert button.isEnabled(), text
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)


def select_catalog_project(controller, name, *, open_workspace=False):
    navigation = controller.shell.navigation_list
    item = controller.shell._navigation_items["projects"]
    QTest.mouseClick(navigation.viewport(), Qt.MouseButton.LeftButton, pos=navigation.visualItemRect(item).center())
    catalog = controller.catalog_page
    model, view = catalog.project_model, catalog.project_list
    index = next(model.index(i, 0) for i in range(model.rowCount()) if model.index(i, 0).data() == name)
    view.setCurrentIndex(index)
    view.scrollTo(index)
    QApplication.processEvents()
    position = view.visualRect(index).center()
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=position)
    if open_workspace:
        QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, pos=position)


def modal(action, fill):
    """Interact with the actual nested dialog event loop, preserving Qt errors."""
    errors = []
    visited = []
    timer = QTimer()
    timer.setInterval(20)
    deadline = QTimer()
    deadline.setSingleShot(True)

    def timed_out():
        errors.append(TimeoutError("UI dialog did not complete within 60 seconds"))
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QDialog):
            dialog.reject()

    deadline.timeout.connect(timed_out)

    def poll():
        dialog = QApplication.activeModalWidget()
        if dialog is None or isinstance(dialog, QMessageBox):
            return
        timer.stop()
        visited.append(dialog.windowTitle())
        try:
            fill(dialog)
        except Exception as exc:
            errors.append(exc)
            dialog.reject()

    timer.timeout.connect(poll)
    timer.start()
    deadline.start(60000)
    try:
        action()
    finally:
        timer.stop()
        deadline.stop()
    assert visited, "The requested UI action did not open its dialog"
    if errors:
        raise errors[0]


def accept_dialog(dialog):
    box = dialog.findChild(QDialogButtonBox)
    button = box.button(QDialogButtonBox.StandardButton.Ok)
    assert button.isEnabled()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)


def node_menu(controller, kind, label, *, representation_id=None, job_id=None):
    graph = controller._layer_dialog.graph
    for lane in graph._snapshot.lanes:
        graph.expand_lane(lane.lane_id)
    node = next(
        item
        for item in graph.graph_scene.items()
        if getattr(item, "node", None) is not None
        and item.node.kind == kind
        and (representation_id is None or item.node.representation_id == representation_id)
        and (job_id is None or item.node.details.get("job_id") == str(job_id))
    )
    graph.ensureVisible(node)
    graph.fitInView(graph.graph_scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
    QApplication.processEvents()
    selected = []
    errors = []
    timer = QTimer()
    timer.setInterval(20)

    def choose():
        menu = QApplication.activePopupWidget()
        if not isinstance(menu, QMenu):
            return
        timer.stop()
        try:
            action = next(action for action in menu.actions() if action.text() == label)
            assert action.isEnabled() and action.isVisible(), (label, action.toolTip())
            selected.append(action.text())
            QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())
        except Exception as exc:
            errors.append(exc)
            menu.close()

    timer.timeout.connect(choose)
    timer.start()
    position = graph.mapFromScene(node.sceneBoundingRect().center())
    try:
        # Send the real context menu event; QTest right-click alone does not
        # produce it consistently on the offscreen Windows platform.
        from PyQt6.QtGui import QContextMenuEvent

        event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, position, graph.viewport().mapToGlobal(position))
        QApplication.sendEvent(graph.viewport(), event)
    finally:
        timer.stop()
    if errors:
        raise errors[0]
    assert selected == [label]


@pytest.fixture
def live_ui(qtbot, tmp_path, monkeypatch):
    config_path = os.environ.get("KRAKEN_UI_TEST_CONFIG")
    if not config_path:
        pytest.skip("KRAKEN_UI_TEST_CONFIG must point to an isolated acceptance server")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    import PyQt6.QtCore as qt_core

    class AcceptanceSettings(QSettings):
        def __init__(self, *_args, **_kwargs):
            super().__init__(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)

    monkeypatch.setattr(qt_core, "QSettings", AcceptanceSettings)
    from kraken_hub import preferences_dialog

    monkeypatch.setattr(preferences_dialog, "program_server_url", lambda *_args: config["server_url"])
    account = preferences_dialog.ProgramAccount(
        "acceptance",
        "local",
        config["username"],
        config["username"],
        config["password"],
        True,
    )
    # Credentials are provided by the private test configuration, without
    # writing the user's Windows keyring.
    monkeypatch.setattr(preferences_dialog, "default_program_account", lambda *_args: account)
    remote = RemoteServerProjectService.authenticate_local(
        config["server_url"],
        username=config["username"],
        password=config["password"],
        data_dir=tmp_path / "client",
    )
    local = EmbeddedProjectService(tmp_path / "local")
    local.identities.save(remote.auth.principal)
    dual = DualCatalogService(local, remote)
    shell = ProjectManagerShell()
    qtbot.addWidget(shell)
    session = DesktopSession(remote.auth.access_token, remote.auth.principal, "")
    controller = DesktopController(
        shell,
        dual,
        session,
        thumbnail_store_uri="memory://",
        plugin_items=build_plugin_inventory(scan_plugin_directory(Path(__file__).resolve().parents[1] / "plugins")),
    )
    controller._agent_timer.stop()
    failures = []
    messages = []
    timer = QTimer(shell)
    timer.setInterval(20)

    def close_messages():
        dialog = QApplication.activeModalWidget()
        if failures and isinstance(dialog, QDialog) and not isinstance(dialog, QMessageBox):
            dialog.reject()
            return
        if not isinstance(dialog, QMessageBox):
            return
        messages.append((dialog.windowTitle(), dialog.text()))
        if dialog.icon() in {QMessageBox.Icon.Warning, QMessageBox.Icon.Critical}:
            failures.append((dialog.windowTitle(), dialog.text()))
        yes = dialog.button(QMessageBox.StandardButton.Yes)
        ok = dialog.button(QMessageBox.StandardButton.Ok)
        later = next((button for button in dialog.buttons() if button.text() == "Позже"), None)
        button = yes or ok or later
        if button is not None:
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    timer.timeout.connect(close_messages)
    timer.start()
    shell.show()
    yield SimpleNamespace(
        controller=controller,
        remote=remote,
        config=config,
        failures=failures,
        messages=messages,
        session=session,
        service=dual,
        tmp_path=tmp_path,
    )
    qtbot.waitUntil(lambda: not controller._background_workers, timeout=30000)
    controller._agent_timer.stop()
    remote.stop_sync()
    if controller._layer_dialog is not None:
        controller._layer_dialog.close()
    # Existing tasks from other acceptance runs do not belong to this fixture.
    # Closing with active tasks is exercised separately below.
    shell.close_guard = None
    shell.close()
    timer.stop()
    assert not failures, failures


def test_server_pipeline_training_recognition_reviews_and_reports_through_ui(live_ui, qtbot, monkeypatch):
    ui = live_ui
    controller = ui.controller
    name = f"UI acceptance {uuid4().hex[:8]}"

    def create(dialog):
        dialog.findChild(QLineEdit, "projectName").setText(name)
        storage = dialog.findChild(QComboBox)
        storage.setCurrentIndex(next(i for i in range(storage.count()) if storage.itemText(i) == "Сервер"))
        accept_dialog(dialog)

    modal(lambda: click(controller.catalog_page, "Создать проект"), create)
    project = next(project for project in ui.remote.list_projects() if project.name == name)
    catalog = controller.catalog_page
    model, view = catalog.project_model, catalog.project_list
    index = next(i for i in range(model.rowCount()) if model.index(i, 0).data() == name)
    view.setCurrentIndex(model.index(index, 0))
    QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, pos=view.visualRect(model.index(index, 0)).center())
    qtbot.waitUntil(lambda: controller._workspace is not None)
    assert str(controller._project_id) == str(project.id)
    controller.shell.layers_action.trigger()
    dialog = controller._layer_dialog
    assert dialog is not None and dialog.isVisible()

    def add_layer(layer_dialog):
        assert isinstance(layer_dialog, LayerCreationDialog)
        layer_dialog.name_edit.setText("Metal l1")
        layer_dialog.disk_source.edit.setText(ui.config["source_directory"])
        QTest.mouseClick(layer_dialog.scan_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: layer_dialog.scan_result is not None, timeout=30000)
        assert layer_dialog.scan_result.total_files == 100
        assert layer_dialog.scan_result.ready
        # Test both server scanning and linking, without copying the originals.
        layer_dialog.tabs.setCurrentWidget(layer_dialog.manual_tab)
        layer_dialog.manual_images.edit.setText(ui.config["source_directory"])
        accept_dialog(layer_dialog)

    # Neither checking server roots nor scanning the source is permitted on the
    # client. Restrict only these paths, leaving client caches and imports usable.
    original_is_dir = Path.is_dir
    source = Path(ui.config["source_directory"])
    workspace = ui.remote.project_workspace(project.id)
    forbidden = {str(source), workspace.source_project_dir, workspace.derived_project_dir}

    def is_dir(path):
        if str(path) in forbidden:
            pytest.fail(f"Hub checked a server-only directory locally: {path}")
        return original_is_dir(path)

    with monkeypatch.context() as patches:
        patches.setattr(Path, "is_dir", is_dir)
        modal(lambda: QTest.mouseClick(dialog.add_layer_button, Qt.MouseButton.LeftButton), add_layer)
    layer = ui.remote.list_layers(project.id)[0]
    representation = ui.remote.list_representations(project.id, layer.id)[0]
    qtbot.waitUntil(lambda: not controller._background_workers, timeout=60000)
    assert len(ui.remote.frame_cells(project.id, layer.id, representation.id)) == 100
    assert not ui.failures
    dialog.close()
    controller._workspace.matrix_view.ensureVisible(controller._workspace.matrix_view.scene_rect_for_frame(1, 1))
    qtbot.waitUntil(
        lambda: any(
            cell.thumbnail is not None and not cell.thumbnail.isNull()
            for cell in controller._workspace.matrix_view._cells.values()
        ),
        timeout=30000,
    )
    (ui.tmp_path / "project.json").write_text(
        json.dumps(
            {
                "project_id": str(project.id),
                "layer_id": str(layer.id),
                "representation_id": str(representation.id),
            }
        ),
        encoding="utf-8",
    )
    # The catalog and layer state survive a fresh authenticated service.
    reopened = RemoteServerProjectService.authenticate_local(
        ui.config["server_url"],
        username=ui.config["username"],
        password=ui.config["password"],
        data_dir=ui.tmp_path / "reopened",
    )
    assert reopened.get_project(project.id).name == name
    assert len(reopened.frame_cells(project.id, layer.id, representation.id)) == 100
    reopened.stop_sync()

    from kraken_agent.runner import PluginProcessSpec, PluginRegistry

    helper = Path(__file__).parent / "helpers" / "gui_plugin_workflow.py"
    registry = PluginRegistry(
        {
            capability: PluginProcessSpec(
                capability,
                (sys.executable, str(helper), mode),
                interactive=True,
            )
            for capability, mode in (
                ("frames.dataset.prepare.v1", "contour"),
                ("frames.vectorize.v1", "contour"),
                ("dataset.model.train.v1", "train"),
            )
        }
    )
    monkeypatch.setattr(controller.agent_runtime, "interactive_registry", lambda _capability: registry)
    controller.shell.layers_action.trigger()
    before = {str(job.id) for job in ui.remote.plugin_jobs()}
    node_menu(controller, "source", "Подготовить выборку в Contour")
    node_menu(controller, "source", "Получить CIF в Contour")
    qtbot.waitUntil(lambda: not controller._background_workers, timeout=600000)
    jobs = [job for job in ui.remote.plugin_jobs() if str(job.id) not in before]
    assert len(jobs) == 2
    assert all(job.state.value == "succeeded" for job in jobs), [(job.capability, job.state, job.error) for job in jobs]
    direct = next(rep for rep in ui.remote.list_representations(project.id, layer.id) if rep.kind.value == "vector")
    assert len(ui.remote.frame_cells(project.id, layer.id, direct.id)) == 100
    node_menu(controller, "dataset", "Обучить модель в NeuralImage")
    qtbot.waitUntil(lambda: not controller._background_workers, timeout=600000)
    training = next(
        job
        for job in ui.remote.plugin_jobs()
        if job.project_id == project.id and job.capability == "dataset.model.train.v1"
    )
    assert training.state.value == "succeeded", training.error
    node_menu(controller, "model", "Распознать исходники")
    qtbot.waitUntil(lambda: not controller._background_workers, timeout=600000)
    recognition = next(
        job
        for job in ui.remote.plugin_jobs()
        if job.project_id == project.id and job.capability == "frames.binary-segment.v2"
    )
    assert recognition.state.value == "succeeded", recognition.error
    binary = next(rep for rep in ui.remote.list_representations(project.id, layer.id) if rep.purpose.value == "binary")
    assert len(ui.remote.frame_cells(project.id, layer.id, binary.id)) == 100
    node_menu(controller, "binary", "Получить CIF в Contour")
    qtbot.waitUntil(lambda: not controller._background_workers, timeout=600000)
    neural_vectors = next(
        rep
        for rep in ui.remote.list_representations(project.id, layer.id)
        if rep.kind.value == "vector" and rep.source_image_representation_id == binary.id
    )
    assert len(ui.remote.frame_cells(project.id, layer.id, neural_vectors.id)) == 100
    assert not ui.failures

    review_and_report(ui, qtbot, monkeypatch, project, layer, representation, binary, neural_vectors, training, name)


def test_reopened_project_reviews_and_reports_through_ui(live_ui, qtbot, monkeypatch):
    state_path = os.environ.get("KRAKEN_UI_REOPEN_PROJECT")
    if not state_path:
        pytest.skip("Set KRAKEN_UI_REOPEN_PROJECT to a completed pipeline's project.json")
    ui = live_ui
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    project = ui.remote.get_project(state["project_id"])
    layer = next(layer for layer in ui.remote.list_layers(project.id) if str(layer.id) == state["layer_id"])
    reps = ui.remote.list_representations(project.id, layer.id)
    binary = next(rep for rep in reps if rep.purpose.value == "binary")
    neural_vectors = next(
        rep for rep in reps if rep.kind.value == "vector" and rep.source_image_representation_id == binary.id
    )
    training = next(
        job
        for job in ui.remote.plugin_jobs()
        if job.project_id == project.id and job.capability == "dataset.model.train.v1"
    )
    catalog = ui.controller.catalog_page
    model, view = catalog.project_model, catalog.project_list
    index = next(i for i in range(model.rowCount()) if model.index(i, 0).data() == project.name)
    view.setCurrentIndex(model.index(index, 0))
    view.scrollTo(model.index(index, 0))
    QApplication.processEvents()
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=view.visualRect(model.index(index, 0)).center())
    QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, pos=view.visualRect(model.index(index, 0)).center())
    qtbot.waitUntil(lambda: ui.controller._workspace is not None)
    select_catalog_project(ui.controller, project.name)
    modal(lambda: click(catalog, "Участники и роли"), lambda dialog: click(dialog, "Закрыть"))
    assert ui.remote.acting_role == "maintainer"
    select_catalog_project(ui.controller, project.name, open_workspace=True)
    ui.controller.shell.layers_action.trigger()
    assert len(ui.remote.frame_cells(project.id, layer.id, neural_vectors.id)) == 100
    assert (
        len(
            [
                batch
                for batch in ui.remote.review_batches()
                if batch.project_id == project.id and batch.state.value == "completed"
            ]
        )
        >= 2
    )
    verify_reports(ui, qtbot, monkeypatch, project)
    recognize_external_model(ui, qtbot, monkeypatch, project, training)
    task_controls(ui, qtbot, project, layer, binary, neural_vectors, training)


def review_and_report(ui, qtbot, monkeypatch, project, layer, representation, binary, neural_vectors, training, name):
    controller = ui.controller
    workspace = controller._workspace
    workspace.image_representation_combo.setCurrentIndex(workspace.image_representation_combo.findData(str(binary.id)))
    workspace.vector_representation_combo.setCurrentIndex(
        workspace.vector_representation_combo.findData(str(neural_vectors.id))
    )
    assert str(workspace.image_representation_combo.currentData()) == str(binary.id)
    assert str(workspace.vector_representation_combo.currentData()) == str(neural_vectors.id)
    assert next(
        rep for rep in ui.remote.list_representations(project.id, layer.id) if rep.id == neural_vectors.id
    ).active

    from kraken_manager.presentation.qt.role_management import RoleManagementDialog
    from kraken_manager.presentation.qt.widgets.frame_matrix import FrameRect

    batches = []
    for number in range(2):
        username = f"ui_corrector_{uuid4().hex[:8]}"
        performer = RemoteServerProjectService.register_local(
            ui.config["server_url"],
            username=username,
            password=secrets.token_urlsafe(24),
            display_name=username,
            data_dir=ui.tmp_path / username,
        )
        # Opening a registered user's catalog resolves its server identity and
        # creates the linked performer, just as a normal Hub login does.
        performer.list_projects()

        # Roles are assigned through the shared graph, rather than the ACL API.
        def assign_role(role_dialog):
            assert isinstance(role_dialog, RoleManagementDialog)
            row = next(
                i
                for i in range(role_dialog.people.count())
                if role_dialog.people.item(i).data(Qt.ItemDataRole.UserRole).id == performer.auth.principal.id
            )
            role_dialog.people.setCurrentRow(row)
            role_dialog.graph.ensureVisible(role_dialog.graph._nodes["corrector"])
            role_dialog.graph.fitInView(
                role_dialog.graph.graph_scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio
            )
            point = role_dialog.graph.mapFromScene(role_dialog.graph._nodes["corrector"].sceneBoundingRect().center())
            QTest.mouseClick(role_dialog.graph.viewport(), Qt.MouseButton.LeftButton, pos=point)
            QTest.mouseClick(role_dialog.assign_button, Qt.MouseButton.LeftButton)
            assert "corrector" in role_dialog._held()
            click(role_dialog, "Закрыть")

        if controller._layer_dialog is not None:
            controller._layer_dialog.close()
        select_catalog_project(controller, name)
        modal(lambda: click(controller.catalog_page, "Участники и роли"), assign_role)
        assert {role.value for role in ui.remote.project_roles(project.id, performer.auth.principal.id)} == {
            "corrector"
        }
        select_catalog_project(controller, name, open_workspace=True)
        workspace = controller._workspace
        workspace.matrix_view.set_selection(FrameRect(1, number * 5 + 1, 10, number * 5 + 5))
        assert len(workspace.matrix_view.selected_coordinates()) == 50
        package_parent = ui.tmp_path / f"review-{number}"
        package_parent.mkdir()
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: str(package_parent))
        assignee = next(
            item for item in ui.remote.list_performers() if item.principal_id == performer.auth.principal.id
        )

        def issue(review_dialog):
            combo = review_dialog.findChild(QComboBox)
            combo.setCurrentIndex(combo.findData(str(assignee.id)))
            choose = next(button for button in review_dialog.findChildren(QPushButton) if button.text() == "Выбрать…")
            QTest.mouseClick(choose, Qt.MouseButton.LeftButton)
            accept_dialog(review_dialog)

        # The matrix menu is opened by a genuine context event on a selected cell.
        def issue_from_menu():
            from PyQt6.QtGui import QContextMenuEvent

            matrix = workspace.matrix_view
            matrix.ensureVisible(matrix.scene_rect_for_frame(1, number * 5 + 1))
            position = matrix.mapFromScene(matrix.scene_rect_for_frame(1, number * 5 + 1).center())
            timer = QTimer()
            timer.setInterval(20)

            def choose_menu():
                menu = QApplication.activePopupWidget()
                if isinstance(menu, QMenu):
                    timer.stop()
                    action = next(
                        action for action in menu.actions() if action.text() == "Отправить выбранное на проверку…"
                    )
                    assert action.isEnabled()
                    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())

            timer.timeout.connect(choose_menu)
            timer.start()
            try:
                QApplication.sendEvent(
                    matrix.viewport(),
                    QContextMenuEvent(
                        QContextMenuEvent.Reason.Mouse, position, matrix.viewport().mapToGlobal(position)
                    ),
                )
            finally:
                timer.stop()

        modal(issue_from_menu, issue)
        batch = next(
            batch
            for batch in ui.remote.review_batches()
            if batch.project_id == project.id and str(batch.assignee_id) == str(assignee.id)
        )
        assert batch.state.value == "issued" and len(batch.items) == 50
        package = next(package_parent.iterdir())
        files = sorted((package / "vectors").glob("*.cif"))
        assert len(files) == 50
        # Simulate a valid returned external package, preserving 45 files.
        for path in files[:5]:
            text = path.read_text(encoding="utf-8")
            assert "\nDF;" in text
            path.write_text(text.replace("\nDF;", "\nP 0 0 10 0 10 10 0 10;\nDF;", 1), encoding="utf-8")

        performer_local = EmbeddedProjectService(ui.tmp_path / f"{username}-local")
        performer_local.identities.save(performer.auth.principal)
        performer_shell = ProjectManagerShell()
        qtbot.addWidget(performer_shell)
        performer_controller = DesktopController(
            performer_shell,
            DualCatalogService(performer_local, performer),
            DesktopSession(performer.auth.access_token, performer.auth.principal, ""),
            thumbnail_store_uri="memory://",
            plugin_items=[],
        )
        performer_controller._agent_timer.stop()
        performer_shell.show()
        select_catalog_project(performer_controller, name, open_workspace=True)
        assert performer_controller._workspace is not None
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: str(package))

        def preview(preview_dialog):
            table = preview_dialog.findChild(QTableWidget)
            assert table.rowCount() == 50
            assert sum(table.item(row, 1).text() == "Изменён" for row in range(50)) == 5
            click(preview_dialog, "Загрузить изменённые файлы")

        modal(performer_shell.review_return_action.trigger, preview)
        returned = next(value for value in ui.remote.review_batches() if value.id == batch.id)
        assert returned.state.value == "awaiting_acceptance"
        work = _my_work_panel(ui.service, ui.session, controller)
        qtbot.addWidget(work)
        work.show()
        table = work.findChild(QTableWidget)
        row = next(
            row
            for row in range(table.rowCount())
            if table.item(row, 0).text() == "Проверка" and table.item(row, 3).text() == username
        )
        click(table.cellWidget(row, 7), "Принять")
        accepted = next(value for value in ui.remote.review_batches() if value.id == batch.id)
        assert accepted.state.value == "completed"
        batches.append(str(batch.id))
        performer.stop_sync()
        performer_shell.close_guard = None
        performer_shell.close()

    verify_reports(ui, qtbot, monkeypatch, project)
    recognize_external_model(ui, qtbot, monkeypatch, project, training)
    task_controls(ui, qtbot, project, layer, binary, neural_vectors, training)


def recognize_external_model(ui, qtbot, monkeypatch, project, training):
    from kraken_manager.presentation.qt.widgets.frame_matrix import FrameRect

    controller = ui.controller
    # Also exercise the source menu's external-model picker on one frame.
    # The model file is a client download, obtained through the server API.
    controller.shell.layers_action.trigger()
    version_id = next(
        event.payload.get("version_id", event.payload.get("artifact_version_id"))
        for event in ui.remote.history(project.id)
        if event.event_type == "ArtifactVersionCreated" and str(event.payload.get("plugin_job_id")) == str(training.id)
    )
    model_path = ui.remote.managed_artifact_path(project.id, version_id)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: (str(model_path), ""))
    controller._workspace.matrix_view.set_selection(FrameRect(1, 1, 1, 1))
    before = {job.id for job in ui.remote.plugin_jobs()}
    node_menu(controller, "source", "Распознать готовой моделью")
    qtbot.waitUntil(lambda: not controller._background_workers, timeout=120000)
    external = [job for job in ui.remote.plugin_jobs() if job.id not in before]
    assert len(external) == 1 and external[0].state.value == "succeeded"


def verify_reports(ui, qtbot, monkeypatch, project):
    statistics = _statistics_panel(ui.service)
    qtbot.addWidget(statistics)
    statistics.show()
    project_filter = statistics.findChild(QComboBox, "statisticsProjectFilter")
    project_filter.setCurrentIndex(project_filter.findData(str(project.id)))
    click(statistics, "Рассчитать")
    summary = statistics.findChild(QTableWidget, "statisticsSummary")
    assert summary.rowCount() > 0
    values = {summary.item(row, 0).text(): summary.item(row, 1).text() for row in range(summary.rowCount())}
    assert values["Возвращено с изменениями"] == "10"
    assert values["Возвращено без изменений"] == "90"
    for extension in ("csv", "xlsx"):
        destination = ui.tmp_path / f"ui-report.{extension}"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args, path=destination: (str(path), ""))
        click(statistics, f"Экспорт {extension.upper()}")
        assert destination.is_file() and destination.stat().st_size > 0
    from openpyxl import load_workbook

    book = load_workbook(ui.tmp_path / "ui-report.xlsx", read_only=True)
    assert len(book.sheetnames) >= 5
    book.close()


def task_controls(ui, qtbot, project, layer, binary, neural_vectors, training):
    controller = ui.controller
    # A queued desktop job is a fixture; cancellation and deletion use buttons.
    queued = ui.remote.submit_plugin_job(
        principal=ui.session.principal,
        gateway=None,
        project_id=project.id,
        layer_id=layer.id,
        source_representation_id=binary.id,
        target_representation_id=neural_vectors.id,
        coordinates=[(1, 1)],
        capability="frames.vectorize.v1",
        parameters={"input_transport": "server", "execution_host": "desktop"},
        idempotency_key=str(uuid4()),
    )
    work = _my_work_panel(ui.service, ui.session, controller)
    qtbot.addWidget(work)
    work.show()
    work_table = work.findChild(QTableWidget)

    def job_row(job_id):
        jobs = ui.service.plugin_jobs()
        return len(ui.service.review_batches()) + next(index for index, job in enumerate(jobs) if job.id == job_id)

    click(work_table.cellWidget(job_row(queued.id), 7), "Отменить")
    assert next(job for job in ui.remote.plugin_jobs() if job.id == queued.id).state.value == "cancelled"
    click(work_table.cellWidget(job_row(queued.id), 7), "Удалить")
    assert all(job.id != queued.id for job in ui.remote.plugin_jobs())
    controller.shell.layers_action.trigger()
    if any(
        node.details.get("job_id") == str(training.id)
        for lane in controller._layer_dialog.graph._snapshot.lanes
        for node in lane.nodes
    ):
        node_menu(controller, "job", "Удалить шаг из pipeline", job_id=training.id)
    assert any(job.id == training.id for job in ui.remote.plugin_jobs())
    assert all(
        node.details.get("job_id") != str(training.id)
        for lane in controller._layer_dialog.graph._snapshot.lanes
        for node in lane.nodes
    )
    assert len(ui.remote.frame_cells(project.id, layer.id, neural_vectors.id)) == 100
    assert not ui.failures
