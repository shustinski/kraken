"""Tabbed login window (domain LDAP + server account + local) driven offscreen.

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
    QLabel,
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
            tabs.setCurrentIndex(2)
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
                tabs.setCurrentIndex(2)
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


def test_domain_submit_warns_https_for_plain_http_non_loopback_url(qapp, tmp_path, monkeypatch, service_with_account):
    """Plain-HTTP non-loopback URL: the HTTPS warning is the first modal,
    configure_remote_local is never called, the dialog stays open afterwards."""
    url = "http://172.20.10.11:8080"
    monkeypatch.setattr(login_parts, "domain_server_status", lambda server, **kw: "enabled")
    monkeypatch.setattr(login_parts, "program_server_url", lambda settings=None: url)

    def forbidden_configure(**_kwargs):
        pytest.fail("configure_remote_local must not be called for a plain-HTTP non-loopback URL")

    monkeypatch.setattr(service_with_account, "configure_remote_local", forbidden_configure)
    seen: dict = {}
    stage = {"name": "fill_and_click"}
    actors: list = []
    timer = QTimer()
    timer.setInterval(25)

    def poll() -> None:
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            if "warning_text" not in seen:
                seen["warning_text"] = dialog.text()
                stage["name"] = "reject_dialog"
                ok = dialog.button(QMessageBox.StandardButton.Ok)

                def close_warning() -> None:
                    if ok is not None:
                        QTest.mouseClick(ok, Qt.MouseButton.LeftButton)

                _post_action(actors, close_warning)
            return
        if not _is_login_dialog(dialog):
            return
        if stage["name"] == "fill_and_click":
            stage["name"] = "expect_warning"
            server_field = dialog.findChild(QLineEdit, "loginDomainServer")
            username_field = dialog.findChild(QLineEdit, "loginDomainUsername")
            password_field = dialog.findChild(QLineEdit, "loginDomainPassword")
            submit = dialog.findChild(QPushButton, "loginDomainSubmit")

            def click_domain() -> None:
                assert server_field is not None and username_field is not None
                assert password_field is not None and submit is not None
                server_field.setText(url)
                username_field.setText("mark")
                password_field.setText("pw")
                QTest.mouseClick(submit, Qt.MouseButton.LeftButton)

            _post_action(actors, click_domain)
        elif stage["name"] == "reject_dialog":
            stage["name"] = "done"

            def reject_login() -> None:
                dialog.reject()

            _post_action(actors, reject_login)

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
    assert session is None
    assert seen.get("warning_text"), "the HTTPS warning must be the first modal after submit"
    assert "требует HTTPS" in seen["warning_text"]
    assert "Неверный доменный пароль" not in seen["warning_text"]


def test_login_window_server_tab_layout(qapp, tmp_path, monkeypatch, service_with_account):
    """Три вкладки: «Доменная» (по умолчанию), «Серверная», «Локальная»; предзаполнение и кнопки серверной вкладки."""
    monkeypatch.setattr(login_parts, "domain_server_status", lambda url, **kw: "enabled")
    monkeypatch.setattr(login_parts, "program_server_url", lambda settings=None: "http://127.0.0.1:8080")
    seen: dict = {}
    actors: list = []
    timer = QTimer()
    timer.setInterval(25)

    def poll() -> None:
        dialog = QApplication.activeModalWidget()
        if not _is_login_dialog(dialog) or seen:
            return
        tabs = dialog.findChild(QTabWidget, "loginTabs")
        if tabs is None:
            return
        sign_in = dialog.findChild(QPushButton, "loginServerSignIn")
        register = dialog.findChild(QPushButton, "loginServerRegister")
        password = dialog.findChild(QLineEdit, "loginServerPassword")
        if sign_in is None or register is None or password is None:
            return
        seen["count"] = tabs.count()
        seen["texts"] = [tabs.tabText(i) for i in range(tabs.count())]
        seen["default"] = tabs.currentIndex()
        seen["address"] = dialog.findChild(QLineEdit, "loginServerAddress").text()
        seen["username"] = dialog.findChild(QLineEdit, "loginServerUsername").text()
        seen["sign_in"] = (type(sign_in).__name__, sign_in.text())
        seen["register"] = (type(register).__name__, register.text())
        seen["domain_status"] = dialog.findChild(QLabel, "loginDomainStatus").text()
        seen["server_status"] = dialog.findChild(QLabel, "loginServerStatus").text()
        seen["password_echo"] = password.echoMode()

        def close() -> None:
            dialog.reject()

        _post_action(actors, close)

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
    assert session is None
    assert seen["count"] == 3
    assert seen["texts"] == ["Доменная учётная запись", "Серверная учётная запись", "Локальная"]
    assert seen["default"] == 0
    assert seen["address"] == "http://127.0.0.1:8080"
    assert seen["username"] == os.environ.get("USERNAME", "")
    assert seen["sign_in"] == ("QPushButton", "Войти")
    assert seen["register"] == ("QPushButton", "Создать и войти")
    assert seen["domain_status"] == "Доменная авторизация включена"
    assert seen["server_status"] == "Доменная авторизация включена"
    assert seen["password_echo"] == QLineEdit.EchoMode.Password


def test_server_sign_in_success_completes_session(qapp, tmp_path, monkeypatch, service_with_account):
    """«Войти» серверной вкладки: configure_remote_local с введёнными данными, сохранение учётных данных,
    переход на вкладку «Локальная» с подсказкой; локальный вход завершает сессию."""
    monkeypatch.setattr(login_parts, "domain_server_status", lambda url, **kw: "enabled")
    monkeypatch.setattr(login_parts, "program_server_url", lambda settings=None: "http://127.0.0.1:8080")
    monkeypatch.setattr(login_parts, "_try_local_autologin", lambda dialog, service: None)
    calls: dict = {}
    monkeypatch.setattr(
        login_parts, "upsert_program_account",
        lambda username, password, **kw: calls.setdefault("upsert", (username, password)),
    )
    monkeypatch.setattr(login_parts, "save_program_server_url", lambda url, **kw: calls.setdefault("url", url))

    def fake_configure(base_url, username, password):
        calls["configure"] = (base_url, username, password)
        return object()

    monkeypatch.setattr(service_with_account, "configure_remote_local", fake_configure)
    seen: dict = {}
    stage = {"name": "server_sign_in"}
    actors: list = []
    timer = QTimer()
    timer.setInterval(25)

    def poll() -> None:
        dialog = QApplication.activeModalWidget()
        if not _is_login_dialog(dialog):
            return
        tabs = dialog.findChild(QTabWidget, "loginTabs")
        if stage["name"] == "server_sign_in":
            stage["name"] = "expect_handoff"
            sign_in = dialog.findChild(QPushButton, "loginServerSignIn")

            def click_sign_in() -> None:
                if sign_in is None:
                    raise AssertionError("loginServerSignIn not found")
                dialog.findChild(QLineEdit, "loginServerUsername").setText("server-user")
                dialog.findChild(QLineEdit, "loginServerPassword").setText("server-pw")
                QTest.mouseClick(sign_in, Qt.MouseButton.LeftButton)

            _post_action(actors, click_sign_in)
        elif stage["name"] == "expect_handoff":
            if tabs.currentIndex() != 2:
                return  # the click actor (20 ms) may not have fired before this poll
            seen["handoff_tab"] = tabs.currentIndex()
            seen["hint_visible"] = dialog.findChild(QLabel, "loginLocalHint").isVisible()
            stage["name"] = "local_login"
            username_field = dialog.findChild(QLineEdit, "loginUsername")
            password_field = dialog.findChild(QLineEdit, "loginPassword")
            box = dialog.findChild(QDialogButtonBox)

            def complete_local() -> None:
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
    assert calls["configure"] == ("http://127.0.0.1:8080", "server-user", "server-pw")
    assert calls["upsert"] == ("server-user", "server-pw")
    assert calls["url"] == "http://127.0.0.1:8080"
    assert seen["handoff_tab"] == 2
    assert seen["hint_visible"] is True


def test_server_sign_in_rejects_bad_credentials(qapp, tmp_path, monkeypatch, service_with_account):
    """401 от configure_remote_local (ValueError): предупреждение про серверную учётную запись,
    пароль очищается, диалог остаётся открытым, локальный вход доступен."""
    monkeypatch.setattr(login_parts, "domain_server_status", lambda url, **kw: "enabled")
    monkeypatch.setattr(login_parts, "program_server_url", lambda settings=None: "http://127.0.0.1:8080")

    def bad_configure(**_kwargs):
        raise ValueError("Учётная запись не найдена или неверный пароль")

    monkeypatch.setattr(service_with_account, "configure_remote_local", bad_configure)
    seen: dict = {}
    stage = {"name": "server_sign_in"}
    actors: list = []
    timer = QTimer()
    timer.setInterval(25)

    def poll() -> None:
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            if "warning_text" not in seen:
                login = seen["login_dialog"]
                seen["warning_text"] = dialog.text()
                seen["password"] = login.findChild(QLineEdit, "loginServerPassword").text()
                ok = dialog.button(QMessageBox.StandardButton.Ok)

                def close_warning() -> None:
                    stage["name"] = "local_login"
                    if ok is not None:
                        QTest.mouseClick(ok, Qt.MouseButton.LeftButton)

                _post_action(actors, close_warning)
            return
        if not _is_login_dialog(dialog):
            return
        seen.setdefault("login_dialog", dialog)
        if stage["name"] == "server_sign_in":
            stage["name"] = "expect_warning"
            sign_in = dialog.findChild(QPushButton, "loginServerSignIn")

            def click_sign_in() -> None:
                if sign_in is None:
                    raise AssertionError("loginServerSignIn not found")
                dialog.findChild(QLineEdit, "loginServerUsername").setText("server-user")
                dialog.findChild(QLineEdit, "loginServerPassword").setText("server-pw")
                QTest.mouseClick(sign_in, Qt.MouseButton.LeftButton)

            _post_action(actors, click_sign_in)
        elif stage["name"] == "local_login":
            username_field = dialog.findChild(QLineEdit, "loginUsername")
            password_field = dialog.findChild(QLineEdit, "loginPassword")
            box = dialog.findChild(QDialogButtonBox)

            def complete_local() -> None:
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
    assert "Неверный пароль или учётная запись не найдена на сервере" in seen["warning_text"]
    assert seen["password"] == ""


def test_server_register_creates_account_then_signs_in(qapp, tmp_path, monkeypatch, service_with_account):
    """«Создать и войти»: register_local (display_name == логин), затем configure_remote_local,
    затем переход на вкладку «Локальная»."""
    monkeypatch.setattr(login_parts, "domain_server_status", lambda url, **kw: "enabled")
    monkeypatch.setattr(login_parts, "program_server_url", lambda settings=None: "http://127.0.0.1:8080")
    monkeypatch.setattr(login_parts, "_try_local_autologin", lambda dialog, service: None)
    calls: dict = {}
    monkeypatch.setattr(
        login_parts, "upsert_program_account",
        lambda username, password, **kw: calls.setdefault("upsert", (username, password)),
    )
    monkeypatch.setattr(login_parts, "save_program_server_url", lambda url, **kw: calls.setdefault("url", url))

    def fake_register(base_url, *, username, password, display_name="", **_kw):
        calls["register"] = (base_url, username, password, display_name)
        return object()

    monkeypatch.setattr(login_parts.RemoteServerProjectService, "register_local", fake_register)

    def fake_configure(base_url, username, password):
        calls["configure"] = (base_url, username, password)
        return object()

    monkeypatch.setattr(service_with_account, "configure_remote_local", fake_configure)
    seen: dict = {}
    stage = {"name": "server_register"}
    actors: list = []
    timer = QTimer()
    timer.setInterval(25)

    def poll() -> None:
        dialog = QApplication.activeModalWidget()
        if not _is_login_dialog(dialog):
            return
        tabs = dialog.findChild(QTabWidget, "loginTabs")
        if stage["name"] == "server_register":
            stage["name"] = "expect_handoff"
            register = dialog.findChild(QPushButton, "loginServerRegister")

            def click_register() -> None:
                if register is None:
                    raise AssertionError("loginServerRegister not found")
                dialog.findChild(QLineEdit, "loginServerUsername").setText("server-user")
                dialog.findChild(QLineEdit, "loginServerPassword").setText("server-pw")
                QTest.mouseClick(register, Qt.MouseButton.LeftButton)

            _post_action(actors, click_register)
        elif stage["name"] == "expect_handoff":
            if tabs.currentIndex() != 2:
                return  # the click actor (20 ms) may not have fired before this poll
            seen["handoff_tab"] = tabs.currentIndex()
            stage["name"] = "local_login"
            username_field = dialog.findChild(QLineEdit, "loginUsername")
            password_field = dialog.findChild(QLineEdit, "loginPassword")
            box = dialog.findChild(QDialogButtonBox)

            def complete_local() -> None:
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
    assert calls["register"] == ("http://127.0.0.1:8080", "server-user", "server-pw", "server-user")
    assert calls["configure"] == ("http://127.0.0.1:8080", "server-user", "server-pw")
    assert calls["upsert"] == ("server-user", "server-pw")
    assert seen["handoff_tab"] == 2


def test_server_register_shows_server_error_as_is(qapp, tmp_path, monkeypatch, service_with_account):
    """Ошибка сервера при регистрации (403 LAN gate) показывается как есть;
    configure_remote_local не вызывается; локальный вход завершает сессию."""
    from kraken_hub.application.remote_error import RemoteServerError

    message = "Создание локальных аккаунтов из внутренней сети запрещено. Используйте доменную учётную запись."

    def forbidden_register(*_args, **_kwargs):
        raise RemoteServerError(message, status=403)

    monkeypatch.setattr(login_parts, "domain_server_status", lambda url, **kw: "enabled")
    monkeypatch.setattr(login_parts, "program_server_url", lambda settings=None: "http://127.0.0.1:8080")
    monkeypatch.setattr(login_parts.RemoteServerProjectService, "register_local", forbidden_register)
    calls: dict = {}
    monkeypatch.setattr(
        service_with_account, "configure_remote_local",
        lambda base_url, username, password: calls.setdefault("configure", (base_url, username, password)),
    )
    seen: dict = {}
    stage = {"name": "server_register"}
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
                    stage["name"] = "local_login"
                    if ok is not None:
                        QTest.mouseClick(ok, Qt.MouseButton.LeftButton)

                _post_action(actors, close_warning)
            return
        if not _is_login_dialog(dialog):
            return
        if stage["name"] == "server_register":
            stage["name"] = "expect_warning"
            register = dialog.findChild(QPushButton, "loginServerRegister")

            def click_register() -> None:
                if register is None:
                    raise AssertionError("loginServerRegister not found")
                dialog.findChild(QLineEdit, "loginServerUsername").setText("server-user")
                dialog.findChild(QLineEdit, "loginServerPassword").setText("server-pw")
                QTest.mouseClick(register, Qt.MouseButton.LeftButton)

            _post_action(actors, click_register)
        elif stage["name"] == "local_login":
            username_field = dialog.findChild(QLineEdit, "loginUsername")
            password_field = dialog.findChild(QLineEdit, "loginPassword")
            box = dialog.findChild(QDialogButtonBox)

            def complete_local() -> None:
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
    assert seen["warning_text"] == message
    assert "configure" not in calls
