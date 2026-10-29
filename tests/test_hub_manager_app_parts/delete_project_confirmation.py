"""test_hub_manager_app: test delete project confirmation."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import pytest
from PyQt6.QtWidgets import (
    QDialog,
    QInputDialog,
    QMessageBox,
)
from kraken_hub.presentation.qt.manager_app import (
    DesktopController,
)
from kraken_manager.presentation.qt import (
    ProjectListItem,
    ProjectManagerShell,
)

def test_delete_project_confirmation_calls_cache_only_service(qapp, monkeypatch, tmp_path: Path) -> None:
    project = SimpleNamespace(id="project-1", name="Preserve files")
    source = tmp_path / "source" / project.name
    derived = tmp_path / "derived" / project.name
    source.mkdir(parents=True)
    derived.mkdir(parents=True)
    deleted = []

    class Service:
        @staticmethod
        def get_project(_project_id):
            return project

        @staticmethod
        def project_workspace(_project_id):
            return SimpleNamespace(
                source_project_dir=str(source),
                derived_project_dir=str(derived),
            )

        @staticmethod
        def delete_project(**kwargs):
            deleted.append(kwargs)

    shell = ProjectManagerShell()
    controller = object.__new__(DesktopController)
    controller.shell = shell
    controller.service = Service()
    controller.session = SimpleNamespace(principal=SimpleNamespace(id="principal-1"))
    controller.thumbnail_store_uri = ""
    controller._project_id = None
    controller._workspace = None
    controller._layer_dialog = None
    refreshed = []
    controller.refresh_projects = lambda: refreshed.append(True)
    controller._error = pytest.fail

    def confirm(dialog: QInputDialog):
        assert str(source) in dialog.labelText()
        assert str(derived) in dialog.labelText()
        dialog.setTextValue(project.name)
        return QDialog.DialogCode.Accepted

    messages = []
    monkeypatch.setattr(QInputDialog, "exec", confirm)
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *_args: messages.append(True),
    )

    controller.delete_project(
        ProjectListItem(project.id, project.name, 2, 2, "Local")
    )

    assert deleted == [
        {
            "principal": controller.session.principal,
            "project": project,
            "confirmation_name": project.name,
        }
    ]
    assert refreshed == [True]
    assert messages == [True]
    assert source.is_dir()
    assert derived.is_dir()
