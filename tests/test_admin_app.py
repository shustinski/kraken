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

    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
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


def test_shift_and_ctrl_selection_deletes_every_chosen_account(qapp, tmp_path: Path, monkeypatch) -> None:
    del qapp
    from PyQt6.QtWidgets import QAbstractItemView, QMessageBox, QTableWidgetSelectionRange

    store = LocalAccountStore(tmp_path / "accounts.sqlite3", ScryptPasswordHasher())
    store.create_account("admin", "Administrator", "secret")
    store.create_account("operator", "Operator", "secret")
    store.create_account("guest", "Guest", "secret")
    window = AdminWindow()
    window._accounts = store
    window._services = _Services()
    window.reload()

    assert window.accounts.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
    assert window.projects.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
    window.accounts.clearSelection()
    window.accounts.setRangeSelected(QTableWidgetSelectionRange(1, 0, 2, 3), True)
    assert window._selected_usernames() == ["guest", "operator"]
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: None)

    window._delete_account()

    assert store.get_by_username("admin") is not None
    assert store.get_by_username("operator") is None
    assert store.get_by_username("guest") is None


def test_selected_project_rows_are_deleted_together(qapp, tmp_path: Path, monkeypatch) -> None:
    del qapp, monkeypatch
    from PyQt6.QtCore import QItemSelection, QItemSelectionModel

    other = "22222222-2222-2222-2222-222222222222"

    class _Projects(_Services):
        def list_projects(self, *, include_archived: bool = False):
            del include_archived
            return [
                {"project_id": PROJECT_ID, "name": "Demo", "state": "active"},
                {"project_id": other, "name": "Other", "state": "active"},
            ]

    store = LocalAccountStore(tmp_path / "accounts.sqlite3", ScryptPasswordHasher())
    store.create_account("admin", "Administrator", "secret")
    window = AdminWindow()
    window._accounts = store
    window._services = _Projects()
    window.reload()
    window.projects.clearSelection()
    model = window.projects.model()
    window.projects.selectionModel().select(
        QItemSelection(model.index(0, 0), model.index(1, 0)),
        QItemSelectionModel.SelectionFlag.ClearAndSelect,
    )
    assert [name for _project_id, name in window._selected_projects()] == ["Demo", "Other"]
    captured: list[list[tuple[str, str]]] = []
    window.deletion_panel.delete_projects = captured.append

    window._delete_project()

    assert captured == [[(PROJECT_ID, "Demo"), (other, "Other")]]


def test_new_project_deletion_ignores_the_previously_selected_request(qapp) -> None:
    del qapp
    window = AdminWindow()
    panel = window.deletion_panel
    panel.rows = [
        {
            "request_id": "old",
            "state": "rejected",
            "project_name": "Old",
            "requested_by": "x",
            "reason": "",
            "requested_at": "",
            "error": "",
        },
        {
            "request_id": "new",
            "state": "pending",
            "project_name": "New",
            "requested_by": "x",
            "reason": "",
            "requested_at": "",
            "error": "",
        },
    ]
    panel._show_rows()
    panel.table.setCurrentCell(0, 0)
    seen: list[str] = []

    def preview(request_id: str) -> dict:
        seen.append(str(request_id))
        raise RuntimeError("stop")

    panel.operation = lambda: type("Operation", (), {"preview": staticmethod(preview)})()
    window._report = lambda *_args: None

    panel.execute(confirmed=True, request_id="new")

    assert seen == ["new"]
    assert panel.table.currentRow() == 1


def test_deleted_requests_are_hidden_and_people_are_shown_by_name(qapp, tmp_path: Path) -> None:
    del qapp
    store = LocalAccountStore(tmp_path / "accounts.sqlite3", ScryptPasswordHasher())
    operator = store.create_account("operator", "Оператор смены", "secret")
    window = AdminWindow()
    window._accounts = store
    window._services = _Services()
    panel = window.deletion_panel
    panel.rows = [
        {
            "request_id": "gone",
            "state": "deleted",
            "project_name": "Gone",
            "requested_by": operator.account_id,
            "reason": "",
            "requested_at": "",
            "error": "",
        },
        {
            "request_id": "open",
            "state": "pending",
            "project_name": "Open",
            "requested_by": operator.account_id,
            "reason": "",
            "requested_at": "",
            "error": "",
        },
    ]
    panel._show_rows()

    assert panel.table.rowCount() == 1
    assert panel.table.item(0, 0).text() == "Open"
    assert panel.table.item(0, 1).text() == "Оператор смены"

    store.set_enabled(operator.account_id, False)
    window._fill_audit()
    assert window.audit.item(0, 2).text() == "Оператор смены"


def test_admin_console_does_not_require_an_administrator_account() -> None:
    from kraken_manager.domain.identity import SystemRole
    from kraken_admin.project_roles import administrator

    actor = administrator(None)

    assert actor.display_name == "Kraken Admin"
    assert SystemRole.SERVER_ADMIN in actor.system_roles


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
