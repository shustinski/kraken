"""test_hub_manager_app: test image representation source."""
from __future__ import annotations
import pytest
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLineEdit,
    QPushButton,
    QWidget,
)
from kraken_hub.infrastructure.credentials import windows as windows_credentials
from kraken_hub.presentation.qt import windows_hello
from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.presentation.qt.manager_app import (
    DesktopController,
    _development_session,
    _login,
)
from kraken_manager.domain.project import RepresentationKind
from kraken_manager.presentation.qt.widgets import ClickableLabel

def test_login_creates_first_account_in_dialog(qapp, monkeypatch, tmp_path) -> None:
    service = EmbeddedProjectService(tmp_path)
    saved = []
    monkeypatch.setattr(windows_credentials, "credentials_available", lambda **_kwargs: False)
    monkeypatch.setattr(windows_credentials, "save_credentials", lambda *values, target: saved.append((values, target)))

    def complete_dialog(dialog: QDialog) -> QDialog.DialogCode:
        assert dialog.objectName() == "initialAccountDialog"
        values = {
            "initialAccountUsername": "operator",
            "initialAccountDisplayName": "Оператор",
            "initialAccountPassword": "",
            "initialAccountPasswordConfirmation": "",
        }
        for object_name, value in values.items():
            field = dialog.findChild(QLineEdit, object_name)
            assert field is not None
            field.setText(value)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", complete_dialog)

    session = _login(None, service)

    assert session is not None
    assert service.has_accounts
    assert session.principal.subject == "operator"
    assert service.login("operator", "") is not None
    assert saved == [(("operator", ""), windows_credentials.credential_target(tmp_path))]

def test_development_session_creates_and_reuses_vscode_account(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("KRAKEN_DEV_AUTO_LOGIN", "1")
    monkeypatch.setenv("KRAKEN_DEV_USERNAME", "vscode")
    monkeypatch.setenv("KRAKEN_DEV_PASSWORD", "")
    service = EmbeddedProjectService(tmp_path)

    created = _development_session(service)
    reopened = _development_session(EmbeddedProjectService(tmp_path))

    assert created is not None and reopened is not None
    assert created.principal.id == reopened.principal.id
    assert created.principal.subject == "vscode"

def test_login_autofills_only_after_windows_verification(qapp, monkeypatch, tmp_path) -> None:
    service = EmbeddedProjectService(tmp_path)
    service.create_initial_account("operator", "Operator", "secret")
    verification_windows = []
    saved = []
    targets = []
    monkeypatch.setattr(windows_credentials, "credentials_available", lambda *, target: targets.append(target) or True)
    monkeypatch.setattr(windows_credentials, "load_credentials", lambda *, target: targets.append(target) or ("operator", "secret"))
    monkeypatch.setattr(
        windows_hello,
        "verify_windows_identity",
        lambda window: verification_windows.append(window) or True,
    )
    monkeypatch.setattr(windows_credentials, "save_credentials", lambda *values, target: saved.append((values, target)))

    def accept_autofilled_dialog(dialog: QDialog) -> QDialog.DialogCode:
        assert dialog.objectName() == "loginDialog"
        username = dialog.findChild(QLineEdit, "loginUsername")
        password = dialog.findChild(QLineEdit, "loginPassword")
        assert username is not None and username.text() == ""
        assert password is not None and password.text() == ""
        assert verification_windows == []
        assert targets == []
        dialog.show()
        autofill = dialog.findChild(QPushButton, "loginAutofill")
        assert autofill is not None
        autofill.click()
        assert username is not None and username.text() == "operator"
        assert password is not None and password.text() == "secret"
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", accept_autofilled_dialog)

    session = _login(None, service)

    assert session is not None
    assert verification_windows and verification_windows[0] != 0
    assert saved == [(("operator", "secret"), windows_credentials.credential_target(tmp_path))]
    assert targets == [windows_credentials.credential_target(tmp_path)] * 2

def test_login_does_not_autofill_when_windows_verification_is_cancelled(qapp, monkeypatch, tmp_path) -> None:
    service = EmbeddedProjectService(tmp_path)
    service.create_initial_account("operator", "Operator", "secret")
    monkeypatch.setattr(windows_credentials, "credentials_available", lambda **_kwargs: True)
    monkeypatch.setattr(windows_hello, "verify_windows_identity", lambda _window: False)
    monkeypatch.setattr(
        windows_credentials,
        "load_credentials",
        lambda **_kwargs: pytest.fail("credentials must not be loaded before verification"),
    )

    def cancel_empty_dialog(dialog: QDialog) -> QDialog.DialogCode:
        dialog.show()
        autofill = dialog.findChild(QPushButton, "loginAutofill")
        assert autofill is not None
        autofill.click()
        username = dialog.findChild(QLineEdit, "loginUsername")
        password = dialog.findChild(QLineEdit, "loginPassword")
        assert username is not None and username.text() == ""
        assert password is not None and password.text() == ""
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", cancel_empty_dialog)

    assert _login(None, service) is None

def test_image_representation_source_picker_fills_selected_folder(qapp, monkeypatch) -> None:
    class ServiceStub:
        @staticmethod
        def get_project(_project_id):
            return object()

        @staticmethod
        def list_layers(_project_id):
            return (type("LayerStub", (), {"id": "layer-1"})(),)

    controller = object.__new__(DesktopController)
    controller.service = ServiceStub()
    workspace = QWidget()
    workspace._selected_layer_id = "layer-1"
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *_args: "C:/images",
    )

    def use_picker_and_cancel(dialog: QDialog) -> QDialog.DialogCode:
        picker = dialog.findChild(ClickableLabel, "representationSourceFolderPicker")
        source = dialog.findChild(QLineEdit, "representationSource")
        assert picker is not None and source is not None
        assert picker.minimumWidth() > 0
        assert picker.text() == "Выбрать папку…"
        picker.clicked.emit()
        assert source.text() == "C:/images"
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", use_picker_and_cancel)

    controller._add_representation(workspace, "project-1", RepresentationKind.IMAGE)
