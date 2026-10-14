"""manager_app: scope content."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from pathlib import Path
from uuid import uuid4
from kraken_manager.domain.artifacts import ArtifactScope
from kraken_manager.domain.identity import Permission

from kraken_hub.manager_app_parts.event_matches_node import _confirm_server_external_path


class ScopeContentMixin:
    def _scope_content(
        self,
        project_id,
        *,
        layer_id=None,
    ) -> dict[str, tuple]:
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

        permission_reader = getattr(self.service, "project_permissions", None)
        permissions = (
            frozenset(Permission)
            if not callable(permission_reader)
            else permission_reader(project_id, self.session.principal)
        )
        list_notes = getattr(self.service, "list_notes", None)
        notes = (
            ()
            if not callable(list_notes)
            else list_notes(project_id, layer_id=layer_id)
        )
        list_principals = getattr(self.service, "list_principals", None)
        principals = {
            str(principal.id): principal.display_name
            for principal in (
                ()
                if not callable(list_principals)
                else list_principals(include_inactive=True)
            )
        }
        wanted_scopes = (
            {
                ArtifactScope.PROJECT_ATTACHMENT,
                ArtifactScope.PROJECT_EXTERNAL_LINK,
            }
            if layer_id is None
            else {
                ArtifactScope.LAYER_ATTACHMENT,
                ArtifactScope.LAYER_EXTERNAL_LINK,
            }
        )
        list_series = getattr(self.service, "list_artifact_series", None)
        available_series = (
            ()
            if not callable(list_series)
            else list_series(
                project_id,
                layer_id=layer_id,
                include_archived=True,
            )
        )
        series_items = tuple(
            series
            for series in available_series
            if series.scope in wanted_scopes
            and (
                (layer_id is None and series.layer_id is None)
                or str(series.layer_id) == str(layer_id)
            )
        )
        files: list[Mapping[str, object]] = []
        versions: list[Mapping[str, object]] = []
        for series in series_items:
            active = self.service.active_artifact_version(project_id, series.id)
            files.append(
                {
                    "name": series.name,
                    "scope": (
                        "Вложение"
                        if series.scope
                        in {
                            ArtifactScope.PROJECT_ATTACHMENT,
                            ArtifactScope.LAYER_ATTACHMENT,
                        }
                        else "Внешняя ссылка"
                    ),
                    "active_version": (
                        "—" if active is None else active.filename
                    ),
                    "archived": series.archived,
                    "_series": series,
                }
            )
            for version in self.service.artifact_versions(project_id, series.id):
                versions.append(
                    {
                        "filename": version.filename,
                        "size": f"{version.size_bytes:,}".replace(",", " ") + " байт",
                        "sha256": version.sha256,
                        "author": principals.get(
                            str(version.author_principal_id),
                            str(version.author_principal_id),
                        ),
                        "created_at": version.created_at.isoformat(),
                        "parent": (
                            "—"
                            if version.parent_version_id is None
                            else str(version.parent_version_id)
                        ),
                        "tool": " ".join(
                            value
                            for value in (version.tool_name, version.tool_version)
                            if value
                        )
                        or "—",
                        "provenance": dict(version.parameters),
                        "active": active is not None and active.id == version.id,
                        "_version": version,
                    }
                )

        def add_note() -> None:
            body, accepted = QInputDialog.getMultiLineText(
                self.shell,
                "Новая заметка",
                "Текст:",
            )
            if not accepted or not body.strip():
                return
            try:
                self.service.create_note(
                    principal=self.session.principal,
                    project_id=project_id,
                    layer_id=layer_id,
                    body=body,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось добавить заметку",
                    str(exc),
                )

        def revise_note() -> None:
            if not notes:
                QMessageBox.information(
                    self.shell,
                    "Нет заметок",
                    "Сначала добавьте заметку.",
                )
                return
            note = max(notes, key=lambda item: (item.recorded_at, item.revision))
            body, accepted = QInputDialog.getMultiLineText(
                self.shell,
                "Новая ревизия заметки",
                "Текст:",
                text=note.body,
            )
            if not accepted or not body.strip() or body.strip() == note.body:
                return
            try:
                self.service.revise_note(
                    principal=self.session.principal,
                    note=note,
                    body=body,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось изменить заметку",
                    str(exc),
                )

        def create_file_series(*, external: bool) -> None:
            source, _selected_filter = QFileDialog.getOpenFileName(
                self.shell,
                "Выберите файл",
            )
            if not source:
                return
            if external and not _confirm_server_external_path(
                self.shell, self.service, project_id, source
            ):
                return
            path = Path(source)
            scope = (
                ArtifactScope.PROJECT_EXTERNAL_LINK
                if layer_id is None and external
                else ArtifactScope.PROJECT_ATTACHMENT
                if layer_id is None
                else ArtifactScope.LAYER_EXTERNAL_LINK
                if external
                else ArtifactScope.LAYER_ATTACHMENT
            )
            try:
                series = self.service.create_artifact_series(
                    principal=self.session.principal,
                    project_id=project_id,
                    layer_id=layer_id,
                    scope=scope,
                    name=path.name,
                    idempotency_key=str(uuid4()),
                )
                add = (
                    self.service.add_external_artifact_version
                    if external
                    else self.service.add_managed_artifact_version
                )
                add(
                    principal=self.session.principal,
                    project_id=project_id,
                    series_id=series.id,
                    source=path,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось добавить файл",
                    str(exc),
                )

        def add_version(row: Mapping[str, object], *, external: bool) -> None:
            series = row.get("_series")
            if series is None:
                return
            source, _selected_filter = QFileDialog.getOpenFileName(
                self.shell,
                "Выберите новую версию",
            )
            if not source:
                return
            if external and not _confirm_server_external_path(
                self.shell, self.service, project_id, source
            ):
                return
            active = self.service.active_artifact_version(project_id, series.id)
            try:
                add = (
                    self.service.add_external_artifact_version
                    if external
                    else self.service.add_managed_artifact_version
                )
                add(
                    principal=self.session.principal,
                    project_id=project_id,
                    series_id=series.id,
                    source=source,
                    parent_version_id=None if active is None else active.id,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось добавить версию",
                    str(exc),
                )

        def rename_series(row: Mapping[str, object]) -> None:
            series = row.get("_series")
            if series is None:
                return
            name, accepted = QInputDialog.getText(
                self.shell,
                "Переименовать файл",
                "Название:",
                text=series.name,
            )
            if not accepted or not name.strip() or name.strip() == series.name:
                return
            try:
                self.service.rename_artifact_series(
                    principal=self.session.principal,
                    series=series,
                    name=name,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось переименовать файл",
                    str(exc),
                )

        def archive_series(row: Mapping[str, object]) -> None:
            series = row.get("_series")
            if series is None or series.archived:
                return
            if QMessageBox.question(
                self.shell,
                "Архивировать файл",
                f"Архивировать логический файл «{series.name}»?",
            ) != QMessageBox.StandardButton.Yes:
                return
            try:
                self.service.archive_artifact_series(
                    principal=self.session.principal,
                    series=series,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось архивировать файл",
                    str(exc),
                )

        def activate_version(row: Mapping[str, object]) -> None:
            version = row.get("_version")
            if version is None or bool(row.get("active")):
                return
            if QMessageBox.question(
                self.shell,
                "Активировать версию",
                "Сделать эту неизменяемую версию активной?",
            ) != QMessageBox.StandardButton.Yes:
                return
            try:
                self.service.activate_artifact_version(
                    principal=self.session.principal,
                    project_id=project_id,
                    series_id=version.series_id,
                    version_id=version.id,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось активировать версию",
                    str(exc),
                )

        def export_or_open(row: Mapping[str, object]) -> None:
            version = row.get("_version")
            if version is None:
                return
            if version.external is not None:
                QDesktopServices.openUrl(QUrl(version.external.uri))
                return
            destination, _selected_filter = QFileDialog.getSaveFileName(
                self.shell,
                "Экспортировать файл",
                version.filename,
            )
            if not destination:
                return
            try:
                self.service.export_managed_artifact(
                    project_id,
                    version,
                    destination,
                )
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось экспортировать файл",
                    str(exc),
                )

        def check_external(row: Mapping[str, object]) -> None:
            version = row.get("_version")
            if version is None or version.external is None:
                QMessageBox.information(
                    self.shell,
                    "Управляемый файл",
                    "Эта версия хранится в базе и не изменяется снаружи.",
                )
                return
            try:
                changed = self.service.external_artifact_changed(version)
            except Exception as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось проверить файл",
                    str(exc),
                )
                return
            QMessageBox.information(
                self.shell,
                "Проверка внешнего файла",
                "Файл изменён или недоступен." if changed else "Файл не изменился.",
            )

        general_actions: list[tuple[str, Callable[[], None]]] = []
        if Permission.ADD_NOTE in permissions:
            general_actions.extend(
                (
                    ("Добавить заметку", add_note),
                    ("Изменить последнюю заметку", revise_note),
                )
            )
        if Permission.IMPORT_ARTIFACT in permissions:
            general_actions.extend(
                (
                    (
                        "Добавить файл в базу",
                        lambda: create_file_series(external=False),
                    ),
                    (
                        "Добавить внешнюю ссылку",
                        lambda: create_file_series(external=True),
                    ),
                )
            )
        file_actions = (
            (
                (
                    "Добавить версию в базу",
                    lambda row: add_version(row, external=False),
                ),
                (
                    "Добавить внешнюю версию",
                    lambda row: add_version(row, external=True),
                ),
                ("Переименовать", rename_series),
                ("Архивировать", archive_series),
            )
            if Permission.IMPORT_ARTIFACT in permissions
            else ()
        )
        version_actions = (
            (
                ("Сделать активной", activate_version),
                ("Открыть / экспортировать", export_or_open),
                ("Проверить внешний файл", check_external),
            )
            if Permission.IMPORT_ARTIFACT in permissions
            else (("Открыть / экспортировать", export_or_open),)
        )
        return {
            "notes": tuple(
                {
                    "revision": note.revision,
                    "body": note.body,
                    "author": principals.get(
                        str(note.author_principal_id),
                        str(note.author_principal_id),
                    ),
                    "recorded_at": note.recorded_at.isoformat(),
                }
                for note in notes
            ),
            "files": tuple(files),
            "versions": tuple(versions),
            "actions": tuple(general_actions),
            "file_actions": file_actions,
            "version_actions": version_actions,
        }
