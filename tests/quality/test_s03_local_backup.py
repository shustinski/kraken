"""S03: local workspace binding and backup actions through the Hub GUI."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QComboBox, QDialogButtonBox, QFileDialog, QLineEdit, QMessageBox, QWidget

from kraken_core.settings import QSettings
from kraken_hub.composition.catalog import build_workspace_service
from kraken_manager.domain.project import GridOrientation
from kraken_manager.infrastructure.migration import BundleVerifier
from tests.helpers.qt_harness import catalog_project_names, click_button, create_project, launch_hub, modal_action, wait_until


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _source_backup(tmp_path: Path) -> tuple[Path, str]:
    service = build_workspace_service(data_dir=tmp_path / "source")
    owner = service.create_initial_account("owner", "Owner", "")
    project = service.create_project(
        principal=owner.principal,
        name="Portable",
        width=2,
        height=1,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key="s03-source",
    )
    backup = tmp_path / "backup"
    service.export_backup(project.id, backup, principal=owner.principal)
    return backup, str(project.id)


def _administration(hub):
    page = hub.window.page("administration")
    return next(widget for widget in page.findChildren(QWidget) if hasattr(widget, "_admin_tasks"))


def test_s03_bind_workspace_attaches_two_roots(qapp, tmp_path: Path, monkeypatch):
    source = tmp_path / "sources"
    derived = tmp_path / "derived"
    source.mkdir()
    derived.mkdir()
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes)
    with launch_hub(tmp_path / "hub") as hub:
        settings = QSettings("Kraken", "KrakenHub")
        settings.remove("workspace/source-root")
        settings.remove("workspace/derived-root")

        def fill(dialog):
            if dialog.objectName() == "workspaceRootsDialog":
                dialog.findChild(QLineEdit, "workspaceSourceRoot").setText(str(source))
                dialog.findChild(QLineEdit, "workspaceDerivedRoot").setText(str(derived))
                dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
                return
            name = dialog.findChild(QLineEdit, "projectName")
            assert name is not None
            name.setText("Two roots")
            profile = next(combo for combo in dialog.findChildren(QComboBox) if combo.findText("Локальный") >= 0)
            profile.setCurrentIndex(profile.findText("Локальный"))
            click_button(dialog, "Создать проект")

        visited = modal_action(lambda: click_button(hub.controller.catalog_page, "Создать проект"), fill)
        assert "Хранилища проектов" in visited
        wait_until(lambda: "Two roots" in catalog_project_names(hub))
        project = next(project for project in hub.service.list_projects() if project.name == "Two roots")
        binding = hub.service.project_workspace(project.id)
        assert binding is not None
        assert Path(binding.source_project_dir).parent == source
        assert Path(binding.derived_project_dir).parent == derived


def test_s03_export_backup_creates_verified_folder(qapp, tmp_path: Path, monkeypatch):
    destination = tmp_path / "backups"
    destination.mkdir()
    notices = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: str(destination))
    monkeypatch.setattr(QMessageBox, "information", lambda _parent, title, detail: notices.append((title, detail)))
    with launch_hub(tmp_path / "hub") as hub:
        project = create_project(hub, "Backed up")
        panel = _administration(hub)
        wait_until(lambda: not panel._admin_tasks.busy)
        click_button(panel, "Обновить каталог")
        wait_until(lambda: not panel._admin_tasks.busy)
        click_button(panel, "Создать резервную копию")
        wait_until(lambda: not panel._admin_tasks.busy and bool(notices))
        backups = list(destination.glob(f"kraken-backup-{project.id}-*"))
        assert len(backups) == 1
        assert (backups[0] / "manifest.json").is_file()
        BundleVerifier().verify(backups[0]).raise_for_errors()
        assert notices[0][0] == "Резервная копия создана"


def test_s03_import_backup_restores_project(qapp, tmp_path: Path, monkeypatch):
    backup, project_id = _source_backup(tmp_path)
    notices = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: str(backup))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "information", lambda _parent, title, detail: notices.append((title, detail)))
    with launch_hub(tmp_path / "target") as hub:
        panel = _administration(hub)
        wait_until(lambda: not panel._admin_tasks.busy)
        click_button(panel, "Восстановить резервную копию")
        wait_until(lambda: not panel._admin_tasks.busy and bool(notices))
        assert notices[0][0] == "Резервная копия восстановлена"
        assert str(hub.service.get_project(project_id).id) == project_id


def test_s03_import_corrupted_backup_rejected(qapp, tmp_path: Path, monkeypatch):
    backup, project_id = _source_backup(tmp_path)
    payload = next(path for path in backup.rglob("*") if path.is_file() and path.name != "manifest.json")
    payload.write_bytes(b"corrupted test payload")
    errors = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: str(backup))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, title, detail: errors.append((title, detail)))
    with launch_hub(tmp_path / "target") as hub:
        panel = _administration(hub)
        wait_until(lambda: not panel._admin_tasks.busy)
        click_button(panel, "Восстановить резервную копию")
        wait_until(lambda: not panel._admin_tasks.busy and bool(errors))
        assert errors[0][0] == "Не удалось восстановить резервную копию"
        assert hub.service.get_project(project_id) is None
