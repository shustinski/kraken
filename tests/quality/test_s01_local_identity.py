"""S01: local account bootstrap and login through the real Hub dialog."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLineEdit, QMessageBox, QTabWidget

import kraken_core.settings as kraken_settings
from kraken_hub.composition.catalog import build_workspace_service
from kraken_hub.presentation.qt.parts import login as login_parts


@pytest.fixture
def local_service(tmp_path, monkeypatch):
    monkeypatch.delenv("KRAKEN_SERVER_URL", raising=False)
    monkeypatch.setattr(login_parts, "_remember_credentials", lambda *args, **kwargs: None)
    monkeypatch.setattr(login_parts, "domain_server_status", lambda *args, **kwargs: "disabled")
    monkeypatch.setattr(login_parts, "program_server_url", lambda *args, **kwargs: "http://127.0.0.1:8080")
    saved_settings = kraken_settings._directory
    kraken_settings.configure_settings_directory(tmp_path / "settings")
    try:
        yield build_workspace_service(data_dir=tmp_path)
    finally:
        kraken_settings._directory = saved_settings


def _drive_dialog(service, action):
    app = QApplication.instance() or QApplication([])
    timer = QTimer()
    timer.setInterval(20)
    seen = []

    def poll():
        dialog = app.activeModalWidget()
        if isinstance(dialog, QDialog) and dialog.objectName() in {"initialAccountDialog", "loginDialog"} and not seen:
            seen.append(dialog.objectName())
            action(dialog)

    timer.timeout.connect(poll)
    timer.start()
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(lambda: app.activeModalWidget().reject() if isinstance(app.activeModalWidget(), QDialog) else None)
    deadline.start(10000)
    try:
        session = login_parts._login(None, service)
    finally:
        timer.stop()
        deadline.stop()
    assert seen, "login dialog did not open"
    return session


def test_s01_first_account_created_and_signed_in(local_service):
    assert not local_service.has_accounts

    def create(dialog):
        dialog.findChild(QLineEdit, "initialAccountUsername").setText("operator")
        dialog.findChild(QLineEdit, "initialAccountDisplayName").setText("Operator")
        dialog.findChild(QLineEdit, "initialAccountPassword").setText("secret")
        dialog.findChild(QLineEdit, "initialAccountPasswordConfirmation").setText("secret")
        QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

    session = _drive_dialog(local_service, create)
    assert session is not None
    assert session.principal.subject == "operator"
    assert local_service.has_accounts
    assert local_service.login("operator", "secret") is not None


def test_s01_second_account_rejected(local_service):
    local_service.create_initial_account("operator", "Operator", "secret")
    with pytest.raises(ValueError, match="A local account already exists"):
        local_service.create_initial_account("another", "Another", "secret")


def test_s01_local_login_accepts_correct_password(local_service):
    local_service.create_initial_account("operator", "Operator", "secret")

    def sign_in(dialog):
        dialog.findChild(QTabWidget, "loginTabs").setCurrentIndex(2)
        dialog.findChild(QLineEdit, "loginUsername").setText("operator")
        dialog.findChild(QLineEdit, "loginPassword").setText("secret")
        QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

    session = _drive_dialog(local_service, sign_in)
    assert session is not None
    assert session.principal.subject == "operator"


def test_s01_local_login_rejects_wrong_password(local_service):
    local_service.create_initial_account("operator", "Operator", "secret")
    app = QApplication.instance() or QApplication([])
    observed = []
    timer = QTimer()
    timer.setInterval(20)
    actors = []

    def schedule(callback):
        actor = QTimer()
        actor.setSingleShot(True)
        actor.timeout.connect(callback)
        actor.start(20)
        actors.append(actor)

    def poll():
        dialog = app.activeModalWidget()
        if isinstance(dialog, QMessageBox) and "warning" not in observed:
            observed.append("warning")
            observed.append(dialog.text())
            schedule(lambda: dialog.accept())
        elif isinstance(dialog, QDialog) and dialog.objectName() == "loginDialog":
            if "submitted" not in observed:
                observed.append("submitted")

                def submit():
                    dialog.findChild(QTabWidget, "loginTabs").setCurrentIndex(2)
                    dialog.findChild(QLineEdit, "loginUsername").setText("operator")
                    dialog.findChild(QLineEdit, "loginPassword").setText("wrong")
                    QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)

                schedule(submit)
            elif "warning" in observed:
                schedule(lambda: dialog.reject())
                timer.stop()

    timer.timeout.connect(poll)
    timer.start()
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(lambda: app.activeModalWidget().reject() if isinstance(app.activeModalWidget(), QDialog) else None)
    deadline.start(10000)
    try:
        session = login_parts._login(None, local_service)
    finally:
        timer.stop()
        deadline.stop()
        for actor in actors:
            actor.stop()
    assert session is None
    assert "Неверные учётные данные или аккаунт временно заблокирован." in observed
