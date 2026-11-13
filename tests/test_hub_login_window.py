"""Tabbed login window (domain LDAP + local) driven offscreen.

The dialog is the real ``kraken_hub.presentation.qt.parts.login._login``; a
25 ms watcher timer observes the active modal inside the dialog's ``exec()``
loop the same way ``tests.helpers.qt_harness._sign_in`` does, and posts its
clicks to a separate single-shot actor timer: a nested modal ``exec()``
(such as the ``QMessageBox.warning`` shown by the domain submit handler)
swallows repeat timeouts of the timer whose slot opened it, so the actor
must be a different timer from the watcher that dismisses the warning. A
single-shot deadline rejects a stuck login dialog so a broken dialog fails
the test instead of hanging the run.
"""
from __future__ import annotations

import os
from collections.abc import Callable

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
)

import kraken_hub.presentation.qt.parts.login as login_parts
from kraken_hub.composition.catalog import build_workspace_service
from kraken_hub.infrastructure.credentials import windows as windows_credentials

_ISOLATED_ENV = ("KRAKEN_SERVER_URL", "KRAKEN_GITLAB_TOKEN", "KRAKEN_SERVER_TOKEN", "KRAKEN_UPDATE_URL")

_TIMEOUT_MS = 20_000


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def service_with_account(tmp_path, monkeypatch):
    """Local store with one pre-created account ("mark"/"pw"); hermetic.

    The remote-server env vars are removed (local composition, as in
    ``qt_harness.launch_hub``) and Credential Manager writes are neutralised
    so the test login never persists into the user's vault.
    """
    for name in _ISOLATED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(windows_credentials, "save_credentials", lambda *_args, **_kwargs: True)
    service = build_workspace_service(data_dir=tmp_path)
    assert service.remote is None
    service.create_initial_account(username="mark", display_name="Mark", password="pw")
    return service


def _post_action(actors: list, action: Callable[[], None]) -> None:
    """Run ``action`` from a dedicated single-shot timer (see module docstring).

    The caller keeps the actor alive in ``actors`` until the test ends: a
    parentless QTimer is destroyed with its last Python reference, so an
    unretained actor would never fire.
    """
    actor = QTimer()
    actor.setSingleShot(True)
    actor.setInterval(20)
    actor.timeout.connect(action)
    actor.start()
    actors.append(actor)


def _start_deadline_reject() -> QTimer:
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.setInterval(_TIMEOUT_MS)

    def give_up() -> None:
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QDialog) and not isinstance(dialog, QMessageBox):
            dialog.reject()

    deadline.timeout.connect(give_up)
    deadline.start()
    return deadline


def _is_login_dialog(dialog) -> bool:
    return isinstance(dialog, QDialog) and not isinstance(dialog, QMessageBox) and dialog.objectName() == "loginDialog"


def test_login_window_domain_tab_default_and_local_login_completes(qapp, tmp_path, monkeypatch, service_with_account):
    """service_with_account: локальный аккаунт уже создан (create_initial_account)."""
    monkeypatch.setattr(login_parts, "domain_server_status", lambda url, **kw: "disabled")
    monkeypatch.setattr(login_parts, "program_server_url", lambda settings=None: "http://127.0.0.1:8080")
    seen = {}
    actors: list = []
    timer = QTimer()
    timer.setInterval(25)

    def poll() -> None:
        dialog = QApplication.activeModalWidget()
        if not _is_login_dialog(dialog) or seen:
            return
        tabs = dialog.findChild(QTabWidget, "loginTabs")
        if tabs is None:
            return  # dialog without tabs: the deadline guard fails the test
        seen["tabs"] = tabs
        seen["default"] = tabs.currentIndex()
        seen["domain_username"] = dialog.findChild(QLineEdit, "loginDomainUsername").text()
        seen["submit_kind"] = type(dialog.findChild(QPushButton, "loginDomainSubmit")).__name__
        username_field = dialog.findChild(QLineEdit, "loginUsername")
        password_field = dialog.findChild(QLineEdit, "loginPassword")
        box = dialog.findChild(QDialogButtonBox)

        def complete_local() -> None:
            tabs.setCurrentIndex(1)
            username_field.setText("mark")
            password_field.setText("pw")
            QTest.mouseClick(box.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

        _post_action(actors, complete_local)

    timer.timeout.connect(poll)
    timer.start()
    deadline = _start_deadline_reject()
    try:
        session = login_parts._login(None, service_with_account)
    finally:
        timer.stop()
        deadline.stop()
        for actor in actors:
            actor.stop()
    assert session is not None
    assert seen["default"] == 0
    assert seen["tabs"].tabText(0) == "Доменная учётная запись"
    assert seen["domain_username"] == os.environ.get("USERNAME", "")
    assert seen["submit_kind"] == "QPushButton"


def test_domain_submit_rejects_when_ldap_disabled(qapp, tmp_path, monkeypatch, service_with_account):
    """Клик по loginDomainSubmit при status 'disabled' — предупреждение, диалог открыт, локальный вход завершает сессию."""
    monkeypatch.setattr(login_parts, "domain_server_status", lambda url, **kw: "disabled")
    monkeypatch.setattr(login_parts, "program_server_url", lambda settings=None: "http://127.0.0.1:8080")
    seen: dict = {}
    stage = {"name": "click_domain"}
    actors: list = []
    timer = QTimer()
    timer.setInterval(25)

    def poll() -> None:
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            if "warning_text" not in seen:
                seen["warning_text"] = dialog.text()
                ok = dialog.button(QMessageBox.StandardButton.Ok)

                def close_warning() -> None:
                    if ok is not None:
                        QTest.mouseClick(ok, Qt.MouseButton.LeftButton)

                _post_action(actors, close_warning)
            return
        if not _is_login_dialog(dialog):
            return
        if stage["name"] == "click_domain":
            stage["name"] = "expect_warning"
            submit = dialog.findChild(QPushButton, "loginDomainSubmit")

            def click_domain() -> None:
                if submit is None:
                    raise AssertionError("loginDomainSubmit not found")
                QTest.mouseClick(submit, Qt.MouseButton.LeftButton)

            _post_action(actors, click_domain)
        elif stage["name"] == "expect_warning":
            stage["name"] = "local_login"
            tabs = dialog.findChild(QTabWidget, "loginTabs")
            username_field = dialog.findChild(QLineEdit, "loginUsername")
            password_field = dialog.findChild(QLineEdit, "loginPassword")
            box = dialog.findChild(QDialogButtonBox)

            def complete_local() -> None:
                tabs.setCurrentIndex(1)
                username_field.setText("mark")
                password_field.setText("pw")
                QTest.mouseClick(box.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

            _post_action(actors, complete_local)

    timer.timeout.connect(poll)
    timer.start()
    deadline = _start_deadline_reject()
    try:
        session = login_parts._login(None, service_with_account)
    finally:
        timer.stop()
        deadline.stop()
        for actor in actors:
            actor.stop()
    assert session is not None
    assert seen.get("warning_text"), "domain submit must warn while LDAP is disabled"
    assert "доменная авторизация не включена" in seen["warning_text"].lower()
