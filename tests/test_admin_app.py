from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

from kraken_manager.domain.identity import ProjectRole
from kraken_manager.infrastructure.auth.local import LocalAccountStore, ScryptPasswordHasher
from kraken_admin.admin_app import AdminWindow
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
    assert window.accounts.item(0, 3).text() == "да"
    assert window.accounts.item(1, 1).text() == "operator"
    assert window.projects.item(0).text() == "Demo"
    assert window.maintainers.item(0, 0).text() == "owner"

    window.accounts.selectRow(1)
    window._set_enabled(False)
    assert store.get_account(operator.account_id).enabled is False
    assert window.accounts.item(1, 2).text() == "отключена"

    revoked: dict[str, str] = {}

    def remember(services, accounts, *, project: str, username: str) -> None:
        del services, accounts
        revoked["project"] = project
        revoked["username"] = username

    monkeypatch.setattr("kraken_admin.admin_app.revoke_maintainer", remember)
    window.maintainers.selectRow(0)
    window._revoke_maintainer()
    assert revoked == {"project": PROJECT_ID, "username": "owner"}


def test_admin_without_a_command_opens_the_window() -> None:
    assert _parser().parse_args([]).command is None
