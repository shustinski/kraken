from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

from kraken_manager.domain.identity import ProjectRole
from kraken_manager.infrastructure.auth.local import LocalAccountStore, ScryptPasswordHasher
from kraken_admin.admin_app import AdminWindow
from kraken_server.configuration import default_local_config_path
from kraken_admin.cli import _parser


PROJECT_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


class _Principal:
    subject = "owner"
    display_name = "Owner"


class _Assignment:
    role = ProjectRole.MAINTAINER
    active = True
    principal_id = PROJECT_ID


class _Identities:
    def assignments_for(self, project_id):
        del project_id
        return (_Assignment(),)

    def get(self, principal_id):
        del principal_id
        return _Principal()


class _Services:
    identities = _Identities()

    def list_projects(self, *, include_archived: bool = False):
        del include_archived
        return [{"project_id": PROJECT_ID, "name": "Demo", "state": "active"}]


def test_admin_window_lists_accounts_projects_and_revokes_maintainer(qapp, tmp_path: Path, monkeypatch) -> None:
    del qapp
    store = LocalAccountStore(tmp_path / "accounts.sqlite3", ScryptPasswordHasher())
    administrator = store.create_account("admin", "Administrator", "secret")
    store.grant_global_role(administrator.account_id, "server_admin")
    operator = store.create_account("operator", "Operator", "secret")
    window = AdminWindow()
    window._accounts = store
    window._services = _Services()
    window.reload()

    assert window.accounts.item(0, 1).text() == "admin"
    assert window.accounts.columnCount() == 4
    assert window.accounts.horizontalHeaderItem(3).text() == "Создан"
    assert window.accounts.item(1, 1).text() == "operator"
    from PyQt6.QtWidgets import QPushButton

    assert window.window.findChild(QPushButton, "adminDeleteProject") is not None
    assert window.projects.item(0).text() == "Demo"
    assert window.maintainers.item(0, 0).text() == "owner"

    window.accounts.selectRow(1)
    window._set_enabled(False)
    assert store.get_account(operator.account_id).enabled is False
    assert window.accounts.item(1, 2).text() == "отключена"

    monkeypatch.setattr("kraken_admin.admin_app._confirm_by_typing", lambda *_args: True)
    window._delete_account()
    assert store.get_by_username("operator") is None
    assert any(row["action"] == "account.deleted" for row in store.administration_audit())

    revoked: dict[str, str] = {}

    def remember(services, accounts, *, project: str, username: str) -> None:
        del services, accounts
        revoked["project"] = project
        revoked["username"] = username

    monkeypatch.setattr("kraken_admin.admin_app.revoke_maintainer", remember)
    window.maintainers.selectRow(0)
    window._revoke_maintainer()
    assert revoked == {"project": PROJECT_ID, "username": "owner"}


def test_deletion_reload_explains_a_missing_schema(qapp, monkeypatch) -> None:
    del qapp
    window = AdminWindow()
    window._services = type("Services", (), {"engine": object()})()
    reported: list[tuple[str, str]] = []
    monkeypatch.setattr(window, "_report", lambda title, exc: reported.append((title, str(exc))))
    monkeypatch.setattr(
        "kraken_admin.deletion_panel.ensure_deletion_schema",
        lambda _engine: (_ for _ in ()).throw(
            RuntimeError(
                '(psycopg.errors.UndefinedTable) отношение "project_deletion_requests" не существует\n'
                "[SQL: SELECT project_deletion_requests.request_id FROM project_deletion_requests]"
            )
        ),
    )
    window.deletion_panel.reload()
    assert reported == [(
        "Заявки на удаление недоступны",
        "Не удалось подготовить таблицу заявок на удаление.\n\n"
        '(psycopg.errors.UndefinedTable) отношение "project_deletion_requests" не существует',
    )]
    assert window.deletion_panel.table.rowCount() == 0


def test_deletion_panel_warns_that_auto_confirm_is_unsafe(qapp) -> None:
    del qapp
    window = AdminWindow()
    assert "Небезопасно" in window.deletion_panel.warning.text()
    assert window.deletion_panel.auto_confirm.text() == "Автоподтверждение всех удалений"
    assert window.deletion_panel.auto_confirm.isChecked() is False
    assert window.deletion_panel.table.columnCount() == 6


def test_last_opened_toml_is_reused(qapp, tmp_path: Path, monkeypatch) -> None:
    del qapp
    from PyQt6.QtCore import QSettings

    ini = tmp_path / "admin.ini"
    monkeypatch.setattr(
        "kraken_admin.admin_app._config_settings",
        lambda: QSettings(str(ini), QSettings.Format.IniFormat),
    )
    monkeypatch.setattr("kraken_server.configuration.ServerConfig.load", lambda path: path)
    monkeypatch.setattr("kraken_admin.admin_app.connect_local_admin", lambda path: (None, None))
    chosen = tmp_path / "custom-server.toml"
    chosen.write_text("host = '127.0.0.1'\n", encoding="utf-8")
    window = AdminWindow()
    window._report = lambda title, exc: None
    window.open_config(chosen)
    assert window.remembered_config_path() == chosen.resolve()

    again = AdminWindow()
    assert again.remembered_config_path() == chosen.resolve()
    chosen.unlink()
    assert again.remembered_config_path() == default_local_config_path()


def test_admin_without_a_command_opens_the_window() -> None:
    assert _parser().parse_args([]).command is None
