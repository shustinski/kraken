from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QDialog, QFileDialog, QLineEdit, QWidget

from kraken_hub import windows_credentials
from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.manager_app import DesktopController, _development_session, _login
from kraken_manager.domain.project import RepresentationKind
from kraken_manager.presentation.qt import (
    LayerPipelineSnapshot,
    PipelineLane,
    PipelineNode,
)
from kraken_manager.presentation.qt.widgets import ClickableLabel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_login_creates_first_account_in_dialog(qapp, monkeypatch, tmp_path) -> None:
    service = EmbeddedProjectService(tmp_path)
    saved = []
    monkeypatch.setattr(windows_credentials, "credentials_available", lambda: False)
    monkeypatch.setattr(windows_credentials, "save_credentials", lambda *values: saved.append(values))

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
    assert saved == [("operator", "")]


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
    monkeypatch.setattr(windows_credentials, "credentials_available", lambda: True)
    monkeypatch.setattr(windows_credentials, "load_credentials", lambda: ("operator", "secret"))
    monkeypatch.setattr(
        windows_credentials,
        "verify_windows_identity",
        lambda window: verification_windows.append(window) or True,
    )
    monkeypatch.setattr(windows_credentials, "save_credentials", lambda *values: saved.append(values))

    def accept_autofilled_dialog(dialog: QDialog) -> QDialog.DialogCode:
        assert dialog.objectName() == "loginDialog"
        username = dialog.findChild(QLineEdit, "loginUsername")
        password = dialog.findChild(QLineEdit, "loginPassword")
        assert username is not None and username.text() == "operator"
        assert password is not None and password.text() == "secret"
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", accept_autofilled_dialog)

    session = _login(None, service)

    assert session is not None
    assert verification_windows and verification_windows[0] != 0
    assert saved == [("operator", "secret")]


def test_login_does_not_autofill_when_windows_verification_is_cancelled(qapp, monkeypatch, tmp_path) -> None:
    service = EmbeddedProjectService(tmp_path)
    service.create_initial_account("operator", "Operator", "secret")
    monkeypatch.setattr(windows_credentials, "credentials_available", lambda: True)
    monkeypatch.setattr(windows_credentials, "verify_windows_identity", lambda _window: False)
    monkeypatch.setattr(
        windows_credentials,
        "load_credentials",
        lambda: pytest.fail("credentials must not be loaded before verification"),
    )

    def cancel_empty_dialog(dialog: QDialog) -> QDialog.DialogCode:
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


def test_external_cif_import_uses_source_from_clicked_pipeline_lane(monkeypatch) -> None:
    controller = object.__new__(DesktopController)
    controller._workspace = object()
    controller._project_id = "project-1"
    missing = PipelineNode("missing-cif", "CIF не получен", "missing")
    controller._pipeline_snapshot = lambda *_args: LayerPipelineSnapshot(
        "project-1",
        "layer-1",
        (
            PipelineLane(
                "source-representation-2",
                "Source 2",
                (
                    PipelineNode("source-representation-2", "Source 2", "source"),
                    missing,
                ),
                (("source-representation-2", "missing-cif"),),
            ),
        ),
    )
    calls = []
    monkeypatch.setattr(
        controller,
        "_add_representation",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    controller._layer_manager_action("layer-1", missing, "add_external_vector")

    assert calls
    assert calls[0][0][2] is RepresentationKind.VECTOR
    assert calls[0][1]["source_image_id"] == "source-representation-2"


def test_contour_vectorize_receives_staged_base_layer_path(tmp_path: Path) -> None:
    representation = SimpleNamespace(
        id="binary-representation-1",
        kind=RepresentationKind.IMAGE,
        source="managed-import",
    )

    class ServiceStub:
        data_dir = tmp_path

        @staticmethod
        def list_representations(_project_id, _layer_id):
            return (representation,)

        @staticmethod
        def frame_cells(_project_id, _layer_id, _representation_id):
            return (
                SimpleNamespace(x=3, y=4, sha256="a" * 64),
            )

        @staticmethod
        def read_project_blob(_project_id, _sha256):
            return b"image"

    controller = object.__new__(DesktopController)
    controller.service = ServiceStub()
    controller._project_id = "project-1"
    node = PipelineNode(
        "binary-representation-1",
        "Binary",
        "binary",
        representation_id="binary-representation-1",
    )

    arguments, parameters = controller._contour_launch_arguments(
        layer_id="layer-1",
        node=node,
        action="vectorize",
        source_representation_id="source-representation-1",
    )

    input_directory = Path(arguments[arguments.index("--input-dir") + 1])
    output_directory = Path(arguments[arguments.index("--output-dir") + 1])
    assert input_directory.is_dir()
    assert (input_directory / "3_4.png").read_bytes() == b"image"
    assert output_directory.is_dir()
    assert parameters["input_representation_id"] == "binary-representation-1"
