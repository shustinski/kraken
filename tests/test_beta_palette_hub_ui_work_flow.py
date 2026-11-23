"""Палитра W3 — серверная выдача работы двумя пользователями, «Моя работа», приёмка, план восстановления (live UI). S12."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialogButtonBox,
    QFileDialog,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QWidget,
)

from kraken_core.plugins import build_plugin_inventory, scan_plugin_directory
from kraken_hub.application.remote_error import RemoteServerError
from kraken_hub.composition import DesktopSession, EmbeddedProjectService
from kraken_hub.composition.catalog import DualCatalogService
from kraken_hub.infrastructure.remote.client import RemoteHttpClient, RemoteServerProjectService
from kraken_hub.presentation.qt import preferences_dialog
from kraken_hub.presentation.qt.manager_app import DesktopController
from kraken_hub.presentation.qt.parts.my_work_panel import _my_work_panel
from kraken_manager.domain.identity import ProjectRole
from kraken_manager.presentation.qt import ProjectManagerShell
from tests.test_beta_palette_hub_ui_layers import _add_layer, _create_server_project, _png_folder, _project
from tests.test_beta_palette_hub_ui_project_cycle import _isolate_settings, _ui_config
from tests.test_live_ui_workflow import accept_dialog, modal


def _register(server_url: str, directory: Path, username: str, display_name: str) -> RemoteServerProjectService:
    from kraken_hub.infrastructure.remote.client import RemoteAuth
    from kraken_hub.infrastructure.remote.parts.principal_server_payload import _principal_from_server_payload

    http = RemoteHttpClient(server_url)
    password = f"Palette-{uuid4().hex}"
    session = http.request(
        "POST",
        "/api/v1/auth/accounts",
        token="register",
        payload={"username": username, "password": password, "display_name": display_name},
    )
    token = session["access_token"]
    principal = _principal_from_server_payload(http.request("GET", "/api/v1/session", token=token))
    return RemoteServerProjectService(
        server_url, auth=RemoteAuth(access_token=token, principal=principal), data_dir=directory, http=http,
    )


@pytest.fixture
def stand(monkeypatch, tmp_path, qtbot):
    config = _ui_config()
    _isolate_settings(monkeypatch, tmp_path)
    owner = RemoteServerProjectService.authenticate_local(
        config["server_url"], username=config["username"], password=config["password"], data_dir=tmp_path / "owner",
    )
    account = preferences_dialog.ProgramAccount(
        "palette-s12", "local", config["username"], config["username"], config["password"], True,
    )
    monkeypatch.setattr(preferences_dialog, "program_server_url", lambda *_args, **_kwargs: config["server_url"])
    monkeypatch.setattr(preferences_dialog, "default_program_account", lambda *_args, **_kwargs: account)
    messages: list[str] = []
    shells: list[ProjectManagerShell] = []
    remotes = [owner]

    def controller_for(remote: RemoteServerProjectService, launched: list | None = None):
        local = EmbeddedProjectService(tmp_path / f"local-{uuid4().hex[:8]}")
        local.identities.save(remote.auth.principal)
        service = DualCatalogService(local, remote)
        shell = ProjectManagerShell()
        qtbot.addWidget(shell)
        shells.append(shell)
        session = DesktopSession(remote.auth.access_token, remote.auth.principal, "")

        def record_launch(metadata, arguments=()):
            if launched is not None:
                launched.append((getattr(metadata, "id", ""), tuple(arguments)))

        controller = DesktopController(
            shell, service, session, thumbnail_store_uri="memory://",
            plugin_items=build_plugin_inventory(scan_plugin_directory(Path(__file__).resolve().parents[1] / "plugins")),
            plugin_launcher=record_launch,
        )
        controller._agent_timer.stop()
        shell.page("my_work").set_content(_my_work_panel(service, session, controller))
        shell.show()
        return controller

    timer = QTimer()
    timer.setInterval(20)

    def close_messages():
        dialog = QApplication.activeModalWidget()
        if not isinstance(dialog, QMessageBox):
            return
        messages.append(f"{dialog.windowTitle()}: {dialog.text()}")
        button = dialog.button(QMessageBox.StandardButton.Yes) or dialog.button(QMessageBox.StandardButton.Ok)
        if button is None and dialog.buttons():
            button = dialog.buttons()[0]
        if button is not None:
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    timer.timeout.connect(close_messages)
    timer.start()
    try:
        yield SimpleNamespace(
            config=config, owner=owner, controller_for=controller_for,
            messages=messages, tmp_path=tmp_path, remotes=remotes,
        )
    finally:
        for remote in remotes:
            remote.stop_sync()
        for shell in shells:
            shell.close_guard = None
            shell.close()
        QApplication.processEvents()
        timer.stop()
        for project in owner.list_projects(include_archived=True):
            if str(project.name).startswith("palette-s12") and project.state.value != "archived":
                owner.archive_project(
                    principal=owner.auth.principal, project=project, idempotency_key=str(uuid4()),
                )


def _work_rows(controller, project_name: str) -> list[str]:
    controller.my_work_refresh()
    QApplication.processEvents()
    table = controller.shell.page("my_work").findChild(QTableWidget)
    return [
        table.item(row, 0).text()
        for row in range(table.rowCount())
        if table.item(row, 1) is not None and table.item(row, 1).text() == project_name
    ]


def _grant(owner, project, principal_id, role: ProjectRole) -> None:
    revision = 0
    try:
        revision = owner.project_role_revision(project.id, principal_id)
    except RemoteServerError as exc:
        if exc.status != 404:
            raise
    owner.assign_project_role(
        principal=owner.auth.principal, project=project, target_principal_id=principal_id,
        role=role, expected_revision=revision, idempotency_key=str(uuid4()),
    )


def test_s12_two_users_issue_work_mywork_accept_reissue(stand, qtbot, monkeypatch):
    owner_ui = stand.controller_for(stand.owner)
    performer = _register(
        stand.config["server_url"], stand.tmp_path / "performer", f"palette-performer-{uuid4().hex[:6]}", "Палитра Корректор",
    )
    stand.remotes.append(performer)
    assert stand.owner.auth.principal.id != performer.auth.principal.id
    launched: list = []
    performer_ui = stand.controller_for(performer, launched)
    name = f"palette-s12-{uuid4().hex[:8]}"
    _create_server_project(owner_ui, name)
    select = __import__("tests.test_live_ui_workflow", fromlist=["select_catalog_project"]).select_catalog_project
    select(owner_ui, name, open_workspace=True)
    qtbot.waitUntil(lambda: owner_ui._workspace is not None, timeout=15000)
    _add_layer(owner_ui, qtbot, _png_folder(stand.tmp_path / "work", ("001.png",)), "Металл")
    project = _project(stand.owner, name)
    layer = stand.owner.list_layers(project.id)[0]
    images = _png_folder(stand.tmp_path / "managed", ("002.png",))
    (images / "002.png").rename(images / "001.png")
    vectors = stand.tmp_path / "vectors"
    vectors.mkdir()
    (vectors / "001.cif").write_bytes(b"DS 1 1 1;\nP 0 0 1 0 1 1;\nDF;\nE\n")

    def fill_source(form, title, folder):
        form.findChild(QLineEdit, "representationName").setText(title)
        form.findChild(QWidget, "representationSourceFolderPicker").setProperty("sourceDirectory", str(folder))
        accept_dialog(form)

    modal(
        owner_ui._workspace.add_image_representation_button.click,
        lambda form: fill_source(form, "Кадры", images),
    )
    qtbot.waitUntil(lambda: not owner_ui._background_workers, timeout=60000)
    modal(
        owner_ui._workspace.add_vector_representation_button.click,
        lambda form: fill_source(form, "Вектор", vectors),
    )
    qtbot.waitUntil(lambda: not owner_ui._background_workers, timeout=60000)

    def frame_ready() -> bool:
        image_id = str(owner_ui._workspace.image_representation_combo.currentData() or "")
        vector_id = str(owner_ui._workspace.vector_representation_combo.currentData() or "")
        if not image_id or not vector_id:
            return False
        image_cells = stand.owner.frame_cells(project.id, layer.id, image_id, bounds=(1, 1, 1, 1))
        vector_cells = stand.owner.frame_cells(project.id, layer.id, vector_id, bounds=(1, 1, 1, 1))
        return bool(image_cells and image_cells[0].artifact_version_id and vector_cells and vector_cells[0].artifact_version_id)

    qtbot.waitUntil(frame_ready, timeout=60000)
    from kraken_core.frame_matrix.qt_parts.viewport_tile_item import FrameSelection

    owner_ui._workspace.matrix_view.set_selection(FrameSelection.single(1, 1))
    _grant(stand.owner, project, performer.auth.principal.id, ProjectRole.CORRECTOR)
    performer_id = next(
        str(person.id) for person in stand.owner.list_performers()
        if str(person.principal_id or "") == str(performer.auth.principal.id)
    )

    def issue(dialog):
        assignee = next(box for box in dialog.findChildren(QComboBox) if box.findData(performer_id) >= 0)
        assignee.setCurrentIndex(assignee.findData(performer_id))
        placement = next(box for box in dialog.findChildren(QComboBox) if box.findText("На сервере") >= 0)
        placement.setCurrentIndex(placement.findText("На сервере"))
        accept_dialog(dialog)

    seen = len(stand.messages)
    modal(owner_ui.issue_work_dialog, issue)
    qtbot.waitUntil(lambda: len(stand.messages) > seen, timeout=15000)
    assert any("Работа выдана" in text or "Папка на сервере" in text for text in stand.messages[seen:]), stand.messages[seen:]
    assert "Исправление векторов" not in _work_rows(owner_ui, name)
    assert "Исправление векторов" in _work_rows(performer_ui, name)
    click_refresh = next(
        button for button in performer_ui.shell.page("my_work").findChildren(QPushButton) if button.text() == "Обновить"
    )
    QTest.mouseClick(click_refresh, Qt.MouseButton.LeftButton)
    table = performer_ui.shell.page("my_work").findChild(QTableWidget)
    row = next(
        index for index in range(table.rowCount())
        if table.item(index, 0) is not None
        and table.item(index, 0).text() == "Исправление векторов"
        and table.item(index, 1) is not None
        and table.item(index, 1).text() == name
    )
    opened = len(stand.messages)
    QTest.mouseClick(
        next(button for button in table.cellWidget(row, 7).findChildren(QPushButton) if button.text() == "Открыть"),
        Qt.MouseButton.LeftButton,
    )
    QApplication.processEvents()
    assert launched or any("Исправление векторов" in text or "не установлен" in text for text in stand.messages[opened:])
    package = Path(next(
        order.destination for order in stand.owner.work_orders()
        if order.kind.value == "vector_review" and Path(order.destination, "kraken-review.json").is_file()
    ))
    corrected = next(package.rglob("*.cif"))
    corrected.write_bytes(corrected.read_bytes() + b"\nP 2 0 3 0 3 2;\n")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args, **_kwargs: str(package))

    def accept_preview(dialog):
        button = next(item for item in dialog.findChildren(QPushButton) if "Загрузить" in item.text())
        assert button.isEnabled()
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    modal(owner_ui.load_review_return, accept_preview)
    assert any(event.event_type == "ReviewBatchAccepted" for event in stand.owner.history(project.id))
    before = len(_work_rows(performer_ui, name))
    issued = len(stand.messages)
    modal(owner_ui.issue_work_dialog, issue)
    qtbot.waitUntil(lambda: len(stand.messages) > issued, timeout=15000)
    assert any("Работа выдана" in text for text in stand.messages[issued:]), stand.messages[issued:]
    assert len(_work_rows(performer_ui, name)) > before


def test_s12_issue_without_folder_rejected(stand, qtbot):
    owner_ui = stand.controller_for(stand.owner)
    name = f"palette-s12-{uuid4().hex[:8]}"
    _create_server_project(owner_ui, name)
    select = __import__("tests.test_live_ui_workflow", fromlist=["select_catalog_project"]).select_catalog_project
    select(owner_ui, name, open_workspace=True)
    qtbot.waitUntil(lambda: owner_ui._workspace is not None, timeout=15000)
    _add_layer(owner_ui, qtbot, _png_folder(stand.tmp_path / "nofolder", ("001.png",)), "Металл")
    before = len(stand.owner.work_orders())

    def issue(dialog):
        accept_dialog(dialog)

    modal(owner_ui.issue_work_dialog, issue)
    assert any("Каталог не выбран" in text for text in stand.messages)
    assert len(stand.owner.work_orders()) == before


def test_s12_review_return_without_folder_rejected(stand, qtbot, monkeypatch):
    owner_ui = stand.controller_for(stand.owner)
    name = f"palette-s12-{uuid4().hex[:8]}"
    _create_server_project(owner_ui, name)
    select = __import__("tests.test_live_ui_workflow", fromlist=["select_catalog_project"]).select_catalog_project
    select(owner_ui, name, open_workspace=True)
    qtbot.waitUntil(lambda: owner_ui._workspace is not None, timeout=15000)
    before = len(stand.owner.review_batches())
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args, **_kwargs: "")
    owner_ui.shell.review_return_action.trigger()
    QApplication.processEvents()
    assert len(stand.owner.review_batches()) == before
    assert not any("Импорт заблокирован" in text for text in stand.messages)


def test_s12_observer_cannot_see_private_project_until_role(stand):
    owner_ui = stand.controller_for(stand.owner)
    name = f"palette-s12-{uuid4().hex[:8]}"
    _create_server_project(owner_ui, name)
    project = _project(stand.owner, name)
    observer = _register(
        stand.config["server_url"], stand.tmp_path / "observer", f"palette-observer-{uuid4().hex[:6]}", "Палитра Наблюдатель",
    )
    stand.remotes.append(observer)
    assert name not in {item.name for item in observer.list_projects()}
    _grant(stand.owner, project, observer.auth.principal.id, ProjectRole.VIEWER)
    assert name in {item.name for item in observer.list_projects()}
    with pytest.raises(RemoteServerError) as caught:
        observer.rename_project(
            principal=observer.auth.principal, project=observer.get_project(project.id),
            name=f"{name}-x", idempotency_key=str(uuid4()),
        )
    assert caught.value.status == 403


def test_s12_work_plan_recovery_setup_and_export(stand, qtbot, monkeypatch):
    owner_ui = stand.controller_for(stand.owner)
    name = f"palette-s12-{uuid4().hex[:8]}"
    _create_server_project(owner_ui, name)
    select = __import__("tests.test_live_ui_workflow", fromlist=["select_catalog_project"]).select_catalog_project
    select(owner_ui, name, open_workspace=True)
    qtbot.waitUntil(lambda: owner_ui._workspace is not None, timeout=15000)
    _add_layer(owner_ui, qtbot, _png_folder(stand.tmp_path / "plan", ("001.png", "002.png")), "Металл")
    owner_ui.shell.work_planning_action.trigger()
    qtbot.waitUntil(lambda: owner_ui._work_planning_dialog is not None and owner_ui._work_planning_dialog.isVisible())
    dialog = owner_ui._work_planning_dialog
    qtbot.waitUntil(lambda: dialog._worker is None and dialog.layer_ids.count() >= 1, timeout=20000)
    dialog.plan_title.setText("План палитры")
    QTest.mouseClick(dialog.create_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog._worker is None and dialog.plans.count() >= 1, timeout=20000)

    def fill_template(editor):
        editor.title.setText("Шаблон палитры")
        editor.step_title.setText("Проверка слоя")
        editor.criterion.setText("Слой просмотрен")
        QTest.mouseClick(editor.add_button, Qt.MouseButton.LeftButton)
        save = editor.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Save)
        QTest.mouseClick(save, Qt.MouseButton.LeftButton)

    modal(dialog.new_template_button.click, fill_template)
    qtbot.waitUntil(lambda: dialog._worker is None and dialog.templates.count() >= 1, timeout=20000)
    metal = next(dialog.layer_ids.item(row) for row in range(dialog.layer_ids.count()) if dialog.layer_ids.item(row).text() == "Металл")
    metal.setCheckState(Qt.CheckState.Checked)
    dialog.template_frames.setText("1,1")
    QTest.mouseClick(dialog.apply_template_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog._worker is None and dialog.items.rowCount() >= 1, timeout=20000)
    QTest.mouseClick(dialog.board_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog._board_dialog is not None and dialog._board_dialog.isVisible())
    board = dialog._board_dialog.board
    qtbot.waitUntil(lambda: board.report is not None, timeout=20000)
    destination = stand.tmp_path / "plans.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args, **_kwargs: (str(destination), "CSV (*.csv)"))
    QTest.mouseClick(board.csv_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: destination.is_file(), timeout=20000)
    exported = destination.read_text(encoding="utf-8")
    assert "Проверка слоя" in exported
