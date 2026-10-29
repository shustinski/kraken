"""manager_app: rename project."""
from __future__ import annotations
import os
import shutil
from pathlib import Path
from uuid import uuid4
from kraken_core.frame_matrix import (
    StoreNamespace,
    StorePolicy,
    ThumbnailStoreFactory,
)
from kraken_manager.presentation.qt.models import ProjectListItem


class RenameProjectMixin:
    def refresh_projects(self) -> None:
        label_for = getattr(self.service, "project_storage_label", None)
        items = [
            ProjectListItem(
                project_id=str(project.id),
                name=project.name,
                width=project.width,
                height=project.height,
                storage_label=(
                    label_for(project.id)
                    if callable(label_for)
                    else "Локальный файл"
                ),
                status=project.state.value,
                archived=project.state.value == "archived",
                metadata={
                    "orientation": project.orientation.value,
                    "revision": project.revision,
                    "storage_profile": project.storage_profile,
                },
            )
            for project in self.service.list_projects(
                include_archived=self.catalog_page.show_archived_check.isChecked()
            )
        ]
        self.catalog_page.project_model.replace_items(items)

    def rename_project(self, item: ProjectListItem | None) -> None:
        if item is None:
            return
        from PyQt6.QtWidgets import QInputDialog, QMessageBox

        project = self.service.get_project(item.project_id)
        if project is None:
            self._error("Проект больше не доступен.")
            self.refresh_projects()
            return
        name, accepted = QInputDialog.getText(
            self.shell,
            "Переименовать проект",
            "Новое название:",
            text=project.name,
        )
        if not accepted or name.strip() == project.name:
            return
        is_remote = bool(getattr(self.service, "is_remote_project", lambda _id: False)(project.id))
        if not is_remote:
            answer = QMessageBox.question(
                self.shell,
                "Подтверждение переименования",
                "Будут согласованно переименованы папки исходных и производных данных. Продолжить?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            renamed = self.service.rename_project(
                principal=self.session.principal,
                project=project,
                name=name,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            self._error(str(exc))
            return
        self.refresh_projects()
        if self._project_id == str(renamed.id) and self._workspace is not None:
            self._workspace.set_project_title(renamed.name)

    def archive_project(self, item: ProjectListItem | None) -> None:
        if item is None:
            return
        from PyQt6.QtWidgets import QMessageBox

        project = self.service.get_project(item.project_id)
        if project is None:
            self._error("Проект больше не доступен")
            return
        answer = QMessageBox.question(
            self.shell,
            "Архивировать проект",
            f"Перевести «{project.name}» в read-only архив?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.service.archive_project(
                principal=self.session.principal,
                project=project,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            self._error(str(exc))
            return
        self.refresh_projects()

    def restore_project(self, item: ProjectListItem | None) -> None:
        if item is None:
            return
        project = self.service.get_project(item.project_id)
        if project is None:
            self._error("Проект больше не доступен")
            return
        try:
            self.service.restore_project(
                principal=self.session.principal,
                project=project,
                idempotency_key=str(uuid4()),
            )
        except Exception as exc:
            self._error(str(exc))
            return
        self.refresh_projects()

    def _clear_configured_thumbnail_cache(self, project_id: str) -> None:
        store_uri = self.thumbnail_store_uri or os.environ.get("KRAKEN_THUMBNAIL_STORE_URI")
        if not store_uri:
            return
        store = ThumbnailStoreFactory().create(store_uri)
        namespace = StoreNamespace(
            plugin="matrix",
            project=str(project_id),
            generation="v1",
        )
        try:
            store.open(namespace, StorePolicy())
            store.clear_namespace()
        finally:
            store.close()
        root = getattr(store, "root", None)
        if root is None:
            return
        safe_root = Path(root).resolve()
        namespace_root = (safe_root / namespace.digest()).resolve()
        if (
            namespace_root.parent == safe_root
            and namespace_root.is_dir()
            and not namespace_root.is_symlink()
        ):
            shutil.rmtree(namespace_root)
