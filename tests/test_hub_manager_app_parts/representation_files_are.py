"""test_hub_manager_app: test representation files are."""
from __future__ import annotations
from threading import Event, get_ident
from time import monotonic, sleep
from types import SimpleNamespace
import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QMainWindow,
    QMessageBox,
    QWidget,
)
from kraken_hub import manager_app
from kraken_hub.manager_app import (
    DesktopController,
)
from kraken_manager.domain.project import RepresentationKind

def test_representation_files_are_saved_without_blocking_qt(qapp, monkeypatch) -> None:
    started = Event()
    release = Event()
    commit_thread_ids = []
    project = SimpleNamespace(id="project-1")
    layer = SimpleNamespace(id="layer-1")
    representation = SimpleNamespace(id="representation-1")
    import_result = SimpleNamespace(versions=(object(), object()))
    plan = SimpleNamespace(
        ready=True,
        issues=(),
        items=(object(), object()),
        total_bytes=123,
        missing_coordinates=0,
    )

    class ServiceStub:
        @staticmethod
        def get_project(_project_id):
            return project

        @staticmethod
        def list_layers(_project_id):
            return (layer,)

        @staticmethod
        def plan_import_directory(**_kwargs):
            return plan

        @staticmethod
        def create_representation(**_kwargs):
            return representation

        @staticmethod
        def commit_managed_import(**_kwargs):
            commit_thread_ids.append(get_ident())
            started.set()
            if not release.wait(3):
                raise TimeoutError("test did not release the import")
            return import_result

    class FieldStub:
        def __init__(self, value):
            self.value = value

        def text(self):
            return self.value

        def toPlainText(self):
            return self.value

        def isChecked(self):
            return bool(self.value)

    class FormStub:
        def __init__(self, parent, _kind):
            self.dialog = parent
            self.name = FieldStub("Исходные изображения")
            self.note = FieldStub("")
            self.active = FieldStub(True)
            self.source_directory = "C:/frames"

        @staticmethod
        def exec():
            return QDialog.DialogCode.Accepted

    shell = QMainWindow()
    workspace = QWidget()
    workspace._selected_layer_id = "layer-1"
    workspace.image_representation_combo = QComboBox(workspace)
    workspace.vector_representation_combo = QComboBox(workspace)
    controller = object.__new__(DesktopController)
    controller.shell = shell
    controller.service = ServiceStub()
    controller.session = SimpleNamespace(principal=object())
    controller._workspace = workspace
    controller._project_id = "project-1"
    controller._layer_dialog = None
    controller._background_workers = set()
    controller._load_representations = lambda *_args: None

    monkeypatch.setattr(manager_app, "RepresentationDialog", FormStub)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args, **_kwargs: pytest.fail("background import must succeed"),
    )
    close_messages = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, title, message, *_args, **_kwargs: close_messages.append(
            (title, message)
        ),
    )

    main_thread_id = get_ident()
    controller._add_representation(
        workspace,
        project.id,
        RepresentationKind.IMAGE,
    )
    try:
        assert started.wait(1)
        assert commit_thread_ids != [main_thread_id]
        assert controller._background_workers
        assert controller._allow_close() is False
        assert close_messages[0][0] == "Сохранение файлов ещё не завершено"

        qt_callback_ran = []
        QTimer.singleShot(0, lambda: qt_callback_ran.append(True))
        qapp.processEvents()
        assert qt_callback_ran == [True]
    finally:
        release.set()
        deadline = monotonic() + 3
        while controller._background_workers and monotonic() < deadline:
            qapp.processEvents()
            sleep(0.01)

    assert not controller._background_workers
    assert shell.statusBar().currentMessage() == "Сохранено файлов: 2."
