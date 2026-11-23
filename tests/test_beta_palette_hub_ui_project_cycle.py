"""Палитра W3 — окно входа и серверный каталог проектов через Hub (офскрин). S10."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen
from uuid import uuid4

import pytest
from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
)

import kraken_hub.presentation.qt.parts.login as login_parts
from kraken_core.plugins import build_plugin_inventory, scan_plugin_directory
from kraken_hub.composition import DesktopSession, EmbeddedProjectService
from kraken_hub.composition.catalog import DualCatalogService, build_workspace_service
from kraken_hub.infrastructure.credentials.secret_store import KeyringSecretStore
from kraken_hub.infrastructure.credentials import windows as windows_credentials
from kraken_hub.infrastructure.remote.client import RemoteServerProjectService
from kraken_hub.presentation.qt.manager_app import DesktopController
from kraken_hub.presentation.qt.parts.connect_saved_server import _connect_saved_server
from kraken_hub.presentation.qt.parts.domain_login import upsert_program_account
from kraken_hub.presentation.qt.preferences_dialog import save_program_server_url
from kraken_manager.presentation.qt import ProjectManagerShell
from tests.test_live_ui_workflow import click, modal, select_catalog_project


def _click(widget, text):
    labels = [button.text() for button in widget.findChildren(QPushButton)]
    if text not in labels:
        raise AssertionError(f"Нет кнопки {text!r}. На странице: {labels}")
    click(widget, text)

BLOCKING = "Неверные учётные данные или аккаунт временно заблокирован."


def _ui_config():
    filename = os.environ.get("KRAKEN_UI_TEST_CONFIG")
    if not filename:
        pytest.skip("Нужен KRAKEN_UI_TEST_CONFIG")
    return json.loads(Path(filename).read_text(encoding="utf-8"))


def _isolate_settings(monkeypatch, tmp_path):
    vault: dict[str, bytes] = {}
    monkeypatch.setattr(KeyringSecretStore, "get", lambda self, key: vault.get(str(key)))
    monkeypatch.setattr(KeyringSecretStore, "set", lambda self, key, value: vault.__setitem__(str(key), value))
    monkeypatch.setattr(KeyringSecretStore, "delete", lambda self, key: vault.pop(str(key), None))
    monkeypatch.setattr(windows_credentials, "save_credentials", lambda *_args, **_kwargs: True)
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    import PyQt6.QtCore as qt_core

    class IsolatedSettings(QSettings):
        def __init__(self, *_args, **_kwargs):
            super().__init__(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)

    monkeypatch.setattr(qt_core, "QSettings", IsolatedSettings)
    monkeypatch.setattr("kraken_core.settings.QSettings", IsolatedSettings)
    import kraken_hub.presentation.qt.preferences_dialog as preferences_dialog

    monkeypatch.setattr(preferences_dialog, "QSettings", IsolatedSettings)
    return IsolatedSettings


def _stand_health(config):
    with urlopen(config["server_url"].rstrip("/") + "/api/v1/health", timeout=5) as response:
        return json.load(response)


def _local_account(service):
    if not service.has_accounts:
        service.create_initial_account(
            username="palette-local", display_name="Палитра", password="palette-local-pw",
        )


def _open_login(service, drive, *, timeout_ms=20000):
    seen: dict = {"warnings": []}
    actors: list = []
    timer = QTimer()
    timer.setInterval(25)
    deadline = QTimer()
    deadline.setSingleShot(True)

    def give_up():
        seen["timeout"] = True
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QDialog):
            dialog.reject()

    def poll():
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            seen["warnings"].append(dialog.text())
            button = dialog.button(QMessageBox.StandardButton.Ok)
            if button is not None:
                QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            return
        if dialog is None or dialog.objectName() != "loginDialog" or seen.get("driven"):
            return
        seen["driven"] = True
        drive(dialog, seen, actors)

    timer.timeout.connect(poll)
    deadline.timeout.connect(give_up)
    timer.start()
    deadline.start(timeout_ms)
    try:
        session = login_parts._login(None, service)
    finally:
        timer.stop()
        deadline.stop()
        for actor in actors:
            actor.stop()
    if seen.get("timeout"):
        pytest.fail(f"Окно входа не завершилось; предупреждения: {seen['warnings']}")
    return session, seen


def test_s10_tabbed_login_local_and_server_tabs_visible_ldap_hidden(monkeypatch, tmp_path, qtbot):
    del qtbot
    config = _ui_config()
    assert _stand_health(config)["domain_auth"] is False
    _isolate_settings(monkeypatch, tmp_path)
    save_program_server_url(config["server_url"])
    service = build_workspace_service(data_dir=tmp_path)
    _local_account(service)

    def drive(dialog, seen, actors):
        tabs = dialog.findChild(QTabWidget, "loginTabs")
        texts = [tabs.tabText(index) for index in range(tabs.count())]
        seen["texts"] = texts
        seen["domain_visible"] = tabs.isTabVisible(0)
        seen["local"] = "Локальная" in texts
        seen["server"] = "Серверная учётная запись" in texts
        dialog.findChild(QLineEdit, "loginServerUsername").setText(config["username"])
        dialog.findChild(QLineEdit, "loginServerPassword").setText(config["password"])
        tabs.setCurrentIndex(1)

        def sign_in():
            QTest.mouseClick(dialog.findChild(QPushButton, "loginServerSignIn"), Qt.MouseButton.LeftButton)

        actor = QTimer()
        actor.setSingleShot(True)
        actor.setInterval(20)
        actor.timeout.connect(sign_in)
        actor.start()
        actors.append(actor)

    session, seen = _open_login(service, drive)
    assert seen["local"] and seen["server"]
    assert seen["domain_visible"] is False
    assert session is not None
    assert session.principal.subject == config["username"]
    assert service.remote is not None
    assert service.remote.base_url.rstrip("/") == config["server_url"].rstrip("/")
    service.remote.list_projects()


def test_s10_saved_server_reconnect_attaches_server_client(monkeypatch, tmp_path, qtbot):
    del qtbot
    config = _ui_config()
    _isolate_settings(monkeypatch, tmp_path)
    save_program_server_url(config["server_url"])
    upsert_program_account(config["username"], config["password"], display_name=config["username"])
    service = build_workspace_service(data_dir=tmp_path)
    _local_account(service)
    local = service.login("palette-local", "palette-local-pw")
    _connect_saved_server(None, service, local.principal)
    assert service.server_connection is not None
    assert service.remote is not None
    assert service.remote.auth.principal.id != local.principal.id
    assert service.remote.base_url.rstrip("/") == config["server_url"].rstrip("/")
    service.remote.list_projects()


def test_s10_wrong_server_credentials_blocked_message(monkeypatch, tmp_path, qtbot):
    del qtbot
    config = _ui_config()
    _isolate_settings(monkeypatch, tmp_path)
    save_program_server_url(config["server_url"])
    service = build_workspace_service(data_dir=tmp_path)
    _local_account(service)

    def drive(dialog, _seen, actors):
        dialog.findChild(QTabWidget, "loginTabs").setCurrentIndex(1)
        dialog.findChild(QLineEdit, "loginServerUsername").setText(config["username"])
        dialog.findChild(QLineEdit, "loginServerPassword").setText("wrong-palette-password")

        def sign_in():
            QTest.mouseClick(dialog.findChild(QPushButton, "loginServerSignIn"), Qt.MouseButton.LeftButton)

        actor = QTimer()
        actor.setSingleShot(True)
        actor.setInterval(20)
        actor.timeout.connect(sign_in)
        actor.start()
        actors.append(actor)
        closer = QTimer()
        closer.setSingleShot(True)
        closer.setInterval(8000)
        closer.timeout.connect(dialog.reject)
        closer.start()
        actors.append(closer)

    _session, seen = _open_login(service, drive)
    assert any(BLOCKING in text for text in seen["warnings"])


@pytest.fixture
def hub(monkeypatch, tmp_path, qtbot):
    config = _ui_config()
    _isolate_settings(monkeypatch, tmp_path)
    remote = RemoteServerProjectService.authenticate_local(
        config["server_url"], username=config["username"], password=config["password"], data_dir=tmp_path / "client",
    )
    local = EmbeddedProjectService(tmp_path / "local")
    local.identities.save(remote.auth.principal)
    service = DualCatalogService(local, remote)
    save_program_server_url(config["server_url"])
    upsert_program_account(config["username"], config["password"], display_name=config["username"])
    QSettings().sync()
    import kraken_hub.presentation.qt.preferences_dialog as preferences_dialog

    account = preferences_dialog.ProgramAccount(
        "palette-s10", "local", config["username"], config["username"], config["password"], True,
    )
    monkeypatch.setattr(preferences_dialog, "program_server_url", lambda *_args, **_kwargs: config["server_url"])
    monkeypatch.setattr(preferences_dialog, "default_program_account", lambda *_args, **_kwargs: account)
    shell = ProjectManagerShell()
    qtbot.addWidget(shell)
    session = DesktopSession(remote.auth.access_token, remote.auth.principal, "")
    controller = DesktopController(
        shell, service, session, thumbnail_store_uri="memory://",
        plugin_items=build_plugin_inventory(scan_plugin_directory(Path(__file__).resolve().parents[1] / "plugins")),
    )
    controller._agent_timer.stop()
    shell.show()
    messages = []
    timer = QTimer(shell)
    timer.setInterval(20)

    def close_messages():
        dialog = QApplication.activeModalWidget()
        if not isinstance(dialog, QMessageBox):
            return
        messages.append(dialog.text())
        button = (
            dialog.button(QMessageBox.StandardButton.Yes)
            or dialog.button(QMessageBox.StandardButton.Ok)
        )
        if button is not None:
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    timer.timeout.connect(close_messages)
    timer.start()
    try:
        yield SimpleNamespace(controller=controller, remote=remote, service=service, messages=messages, tmp_path=tmp_path)
    finally:
        controller._agent_timer.stop()
        shell.close_guard = None
        shell.close()
        QApplication.processEvents()
        timer.stop()
        for project in remote.list_projects(include_archived=True):
            if str(project.name).startswith("palette-s10") and project.state.value != "archived":
                remote.archive_project(
                    principal=remote.auth.principal, project=project, idempotency_key=str(uuid4()),
                )


def test_s10_project_cycle_through_ui(hub, qtbot):
    controller = hub.controller
    name = f"palette-s10-{uuid4().hex[:8]}"
    out_of_band = f"palette-s10b-{uuid4().hex[:8]}"

    def fill_create(dialog):
        dialog.findChild(QLineEdit, "projectName").setText(name)
        profiles = dialog.findChild(QComboBox)
        profiles.setCurrentIndex(profiles.findText("Сервер"))
        box = dialog.findChild(QDialogButtonBox)
        QTest.mouseClick(box.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

    modal(lambda: _click(controller.catalog_page, "Создать проект"), fill_create)
    qtbot.waitUntil(lambda: name in _names(controller), timeout=15000)

    hub.remote.create_project(
        principal=hub.remote.auth.principal,
        name=out_of_band,
        width=2,
        height=1,
        orientation=_orientation(),
        idempotency_key=str(uuid4()),
    )
    _click(controller.catalog_page, "Обновить")
    qtbot.waitUntil(lambda: out_of_band in _names(controller), timeout=15000)

    select_catalog_project(controller, name, open_workspace=True)
    qtbot.waitUntil(lambda: controller._workspace is not None, timeout=15000)
    assert controller.shell.layers_action.isEnabled()
    assert controller.shell.actions_menu.isEnabled()

    renamed = name + "-renamed"

    def fill_rename(dialog):
        dialog.findChild(QLineEdit).setText(renamed)
        dialog.accept()

    select_catalog_project(controller, name)
    modal(lambda: _click(controller.catalog_page, "Переименовать"), fill_rename)
    qtbot.waitUntil(lambda: renamed in _names(controller), timeout=15000)

    select_catalog_project(controller, renamed)
    _click(controller.catalog_page, "В архив")
    qtbot.waitUntil(lambda: renamed not in _names(controller), timeout=15000)
    controller.catalog_page.show_archived_check.setChecked(True)
    _click(controller.catalog_page, "Обновить")
    qtbot.waitUntil(lambda: renamed in _names(controller), timeout=15000)
    select_catalog_project(controller, renamed)
    _click(controller.catalog_page, "Восстановить")
    controller.catalog_page.show_archived_check.setChecked(False)
    _click(controller.catalog_page, "Обновить")
    qtbot.waitUntil(lambda: renamed in _names(controller), timeout=15000)

    def fill_properties(dialog):
        label = dialog.findChild(QLabel)
        text = dialog.windowTitle() if label is None else f"{dialog.windowTitle()} {label.text()}"
        assert renamed in text
        dialog.accept()

    select_catalog_project(controller, renamed)
    modal(lambda: _click(controller.catalog_page, "Свойства"), fill_properties)

    def fill_delete(dialog):
        dialog.findChild(QLineEdit, "deletionRequestReason").setText("палитра")
        box = dialog.findChild(QDialogButtonBox)
        QTest.mouseClick(box.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

    delete_label = (
        "Запросить удаление"
        if any(button.text() == "Запросить удаление" for button in controller.catalog_page.findChildren(QPushButton))
        else "Удалить"
    )
    modal(lambda: _click(controller.catalog_page, delete_label), fill_delete)
    project_id = next(project.id for project in hub.remote.list_projects(include_archived=True) if project.name == renamed)
    qtbot.waitUntil(lambda: bool(hub.remote.project_deletion_requests(project_id)), timeout=15000)
    assert renamed in _names(controller)
    for project in hub.remote.list_projects(include_archived=True):
        if project.name in {renamed, out_of_band} and project.state.value != "archived":
            hub.remote.archive_project(
                principal=hub.remote.auth.principal, project=project, idempotency_key=str(uuid4()),
            )


def test_s10_local_delete_rejects_partial_name_then_removes_project(hub, qtbot):
    controller = hub.controller
    name = f"palette-s10-local-{uuid4().hex[:8]}"
    source = hub.tmp_path / "source-root"
    derived = hub.tmp_path / "derived-root"
    source.mkdir()
    derived.mkdir()
    from kraken_core.settings import QSettings as IsolatedSettings

    settings = IsolatedSettings("Kraken", "KrakenHub")
    settings.setValue("workspace/source-root", str(source))
    settings.setValue("workspace/derived-root", str(derived))
    settings.sync()

    def fill_create(dialog):
        dialog.findChild(QLineEdit, "projectName").setText(name)
        profiles = dialog.findChild(QComboBox)
        profiles.setCurrentIndex(profiles.findText("Локальный"))
        box = dialog.findChild(QDialogButtonBox)
        QTest.mouseClick(box.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

    modal(lambda: _click(controller.catalog_page, "Создать проект"), fill_create)
    qtbot.waitUntil(lambda: name in _names(controller), timeout=15000)

    def fill_name(dialog, value):
        dialog.findChild(QLineEdit).setText(value)
        dialog.accept()

    select_catalog_project(controller, name)
    modal(lambda: _click(controller.catalog_page, "Удалить"), lambda dialog: fill_name(dialog, name[:4]))
    qtbot.waitUntil(
        lambda: any("не совпадает с названием проекта" in text for text in hub.messages),
        timeout=5000,
    )
    assert name in _names(controller)

    select_catalog_project(controller, name)
    modal(lambda: _click(controller.catalog_page, "Удалить"), lambda dialog: fill_name(dialog, name))
    qtbot.waitUntil(lambda: name not in _names(controller), timeout=15000)


def _orientation():
    from kraken_manager.domain.project import GridOrientation

    return GridOrientation.Y_DOWN


def _names(controller):
    model = controller.catalog_page.project_model
    return {model.index(row, 0).data() for row in range(model.rowCount())}
