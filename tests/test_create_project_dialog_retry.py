"""The create-project dialog keeps one idempotency key across a failed attempt.

A new key on every click misses the journal of the attempt that already
created the folder, so the user sees "Папка проекта уже существует".
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import QApplication, QLineEdit, QMessageBox

from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.presentation.qt import manager_app
from kraken_hub.presentation.qt.manager_app import DesktopController
from kraken_manager.presentation.qt import ProjectManagerShell


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _accept_dialog(name: str) -> None:
    def run() -> None:
        dialog = QApplication.activeModalWidget()
        field = None if dialog is None else dialog.findChild(QLineEdit, "projectName")
        if field is None:
            QTimer.singleShot(10, run)
            return
        field.setText(name)
        dialog.accept()

    QTimer.singleShot(0, run)


def test_dialog_reuses_the_idempotency_key_until_the_project_identity_changes(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *_args, **_kwargs: QMessageBox.StandardButton.Ok),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *_args, **_kwargs: QMessageBox.StandardButton.Ok),
    )

    class FakeRuntime:
        def __init__(self, *_args, **_kwargs) -> None:
            self.process = None
            self.base_url = ""
            self.token = ""

        def ensure_started(self):
            return frozenset()

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(manager_app, "LocalAgentRuntime", FakeRuntime)
    settings_path = tmp_path / "hub.ini"
    sources = tmp_path / "sources"
    derived = tmp_path / "derived"
    sources.mkdir()
    derived.mkdir()
    isolated = QSettings(str(settings_path), QSettings.Format.IniFormat)
    isolated.setValue("workspace/source-root", str(sources))
    isolated.setValue("workspace/derived-root", str(derived))
    monkeypatch.setattr(
        "kraken_core.settings.QSettings",
        lambda *_args, **_kwargs: QSettings(str(settings_path), QSettings.Format.IniFormat),
    )

    service = EmbeddedProjectService(tmp_path / "data")
    session = service.create_initial_account("owner", "Owner", "secret")
    keys: list[str] = []
    real_create = service.create_project

    def spy(**kwargs):
        keys.append(kwargs["idempotency_key"])
        if len(keys) == 1:
            QTimer.singleShot(0, lambda: _accept_dialog("Аудит"))
            raise FileExistsError("Папка проекта «Аудит» уже существует.")
        return real_create(**kwargs)

    service.create_project = spy
    shell = ProjectManagerShell()
    controller = DesktopController(shell, service, session, thumbnail_store_uri="memory://", plugin_items=[])
    controller._agent_timer.stop()
    _accept_dialog("Аудит")
    controller.create_project()
    _accept_dialog("Другой")
    controller.create_project()
    controller._agent_timer.stop()

    assert keys[0] == keys[1]
    assert keys[2] != keys[0]
    projects = service.list_projects()
    assert {project.name for project in projects} == {"Аудит", "Другой"}
