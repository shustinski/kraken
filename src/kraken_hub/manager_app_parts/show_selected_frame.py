"""manager_app: show selected frame."""
from __future__ import annotations
from collections.abc import Mapping
from uuid import uuid4
from kraken_manager.domain.identity import Permission
from kraken_manager.presentation.qt import (
    ObjectHistoryEntry,
    ObjectPropertiesSnapshot,
)

from kraken_hub.manager_app_parts.event_matches_node import (
    _confirm_server_external_path,
    _event_payload,
    _history_entries,
)


class ShowSelectedFrameMixin:
    def show_selected_frame(self) -> None:
        workspace = self._workspace
        project_id = str(self._project_id or "")
        if workspace is None or not project_id:
            return
        coordinates = workspace.matrix_view.selected_coordinates(maximum=2)
        if len(coordinates) != 1:
            self._error("Для просмотра карточки выберите ровно один кадр.")
            return
        x, y = coordinates[0]
        cell = workspace.matrix_view.cell_data(x, y)
        payload = cell.payload if isinstance(cell.payload, Mapping) else {}
        frame_id = str(payload.get("frame_id") or payload.get("key") or "")
        project = self.service.get_project(project_id)
        permissions = self.service.project_permissions(
            project_id,
            self.session.principal,
        )
        layer_id = str(getattr(workspace, "_selected_layer_id", ""))
        layer = next(
            (item for item in self.service.list_layers(project_id) if str(item.id) == layer_id),
            None,
        )
        representations = self.service.list_representations(project_id, layer_id)
        image_id = str(workspace.image_representation_combo.currentData() or "")
        vector_id = str(workspace.vector_representation_combo.currentData() or "")
        image = next((item for item in representations if str(item.id) == image_id), None)
        vector = next((item for item in representations if str(item.id) == vector_id), None)
        events = tuple(
            event
            for event in self.service.history(project_id)
            if frame_id and frame_id in str(_event_payload(event))
        )
        principals = {
            str(principal.id): principal.display_name
            for principal in self.service.list_principals(include_inactive=True)
        }
        series_values = tuple(
            series
            for series in self.service.list_artifact_series(
                project_id,
                layer_id=layer_id,
                include_archived=True,
            )
            if str(series.frame_id or "") == frame_id
        )
        notes = self.service.list_notes(
            project_id,
            layer_id=layer_id,
            frame_id=frame_id,
        )
        files: list[Mapping[str, object]] = []
        versions: list[Mapping[str, object]] = []
        for series in series_values:
            active = self.service.active_artifact_version(project_id, series.id)
            files.append(
                {
                    "_series": series,
                    "name": series.name,
                    "scope": series.scope.value,
                    "active_version": None if active is None else str(active.id),
                    "archived": series.archived,
                }
            )
            for version in self.service.artifact_versions(project_id, series.id):
                if version.size_bytes < 1024:
                    size = f"{version.size_bytes} байт"
                elif version.size_bytes < 1024 * 1024:
                    size = f"{version.size_bytes / 1024:.1f} КБ"
                else:
                    size = f"{version.size_bytes / (1024 * 1024):.1f} МБ"
                versions.append(
                    {
                        "_version": version,
                        "filename": version.filename,
                        "size": size,
                        "sha256": version.sha256,
                        "author": principals.get(
                            str(version.author_principal_id),
                            str(version.author_principal_id),
                        ),
                        "created_at": version.created_at.isoformat(),
                        "parent": (
                            None
                            if version.parent_version_id is None
                            else str(version.parent_version_id)
                        ),
                        "tool": " ".join(
                            value
                            for value in (version.tool_name, version.tool_version)
                            if value
                        ),
                        "provenance": dict(version.parameters),
                        "active": active is not None and active.id == version.id,
                    }
                )

        def add_note() -> None:
            from PyQt6.QtWidgets import QInputDialog, QMessageBox

            body, accepted = QInputDialog.getMultiLineText(
                self.shell,
                "Новая заметка",
                "Текст:",
            )
            if not accepted:
                return
            try:
                self.service.create_note(
                    principal=self.session.principal,
                    project_id=project_id,
                    layer_id=layer_id,
                    frame_id=frame_id,
                    body=body,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(self.shell, "Не удалось создать заметку", str(exc))
                return
            QMessageBox.information(
                self.shell,
                "Заметка создана",
                "Заметка сохранена новой неизменяемой ревизией. Переоткройте карточку.",
            )

        def revise_latest_note() -> None:
            from PyQt6.QtWidgets import QInputDialog, QMessageBox

            if not notes:
                QMessageBox.information(self.shell, "Нет заметок", "Сначала создайте заметку.")
                return
            current = max(notes, key=lambda value: (value.recorded_at, value.revision))
            body, accepted = QInputDialog.getMultiLineText(
                self.shell,
                "Изменить заметку",
                "Новая ревизия:",
                text=current.body,
            )
            if not accepted:
                return
            try:
                self.service.revise_note(
                    principal=self.session.principal,
                    note=current,
                    body=body,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(self.shell, "Не удалось изменить заметку", str(exc))
                return
            QMessageBox.information(
                self.shell,
                "Ревизия сохранена",
                "Предыдущая ревизия осталась в истории.",
            )

        def add_version(row: Mapping[str, object], *, external: bool) -> None:
            from PyQt6.QtWidgets import QFileDialog, QMessageBox

            series = row.get("_series")
            if series is None:
                return
            source, _selected_filter = QFileDialog.getOpenFileName(
                self.shell,
                "Выберите новую версию файла",
            )
            if not source:
                return
            if external and not _confirm_server_external_path(
                self.shell, self.service, project_id, source
            ):
                return
            try:
                method = (
                    self.service.add_external_artifact_version
                    if external
                    else self.service.add_managed_artifact_version
                )
                method(
                    principal=self.session.principal,
                    project_id=project_id,
                    series_id=series.id,
                    source=source,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(self.shell, "Не удалось добавить версию", str(exc))
                return
            QMessageBox.information(
                self.shell,
                "Версия добавлена",
                "Новая неизменяемая версия сохранена. Переоткройте карточку.",
            )

        def rename_series(row: Mapping[str, object]) -> None:
            from PyQt6.QtWidgets import QInputDialog, QMessageBox

            series = row.get("_series")
            if series is None:
                return
            name, accepted = QInputDialog.getText(
                self.shell,
                "Переименовать файл",
                "Логическое название:",
                text=series.name,
            )
            if not accepted:
                return
            try:
                self.service.rename_artifact_series(
                    principal=self.session.principal,
                    series=series,
                    name=name,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(self.shell, "Не удалось переименовать файл", str(exc))

        def archive_series(row: Mapping[str, object]) -> None:
            from PyQt6.QtWidgets import QMessageBox

            series = row.get("_series")
            if series is None:
                return
            if QMessageBox.question(
                self.shell,
                "Архивировать файл",
                f"Архивировать логическую серию «{series.name}»?",
            ) != QMessageBox.StandardButton.Yes:
                return
            try:
                self.service.archive_artifact_series(
                    principal=self.session.principal,
                    series=series,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                QMessageBox.warning(self.shell, "Не удалось архивировать файл", str(exc))

        def activate_version(row: Mapping[str, object]) -> None:
            from PyQt6.QtWidgets import QMessageBox

            version = row.get("_version")
            if version is None or bool(row.get("active")):
                return
            if QMessageBox.question(
                self.shell,
                "Активировать прежнюю версию",
                "Сделать выбранную неизменяемую версию активной? История не будет переписана.",
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
                QMessageBox.warning(self.shell, "Не удалось активировать версию", str(exc))

        def export_or_open_version(row: Mapping[str, object]) -> None:
            from PyQt6.QtCore import QUrl
            from PyQt6.QtGui import QDesktopServices
            from PyQt6.QtWidgets import QFileDialog, QMessageBox

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
                QMessageBox.warning(self.shell, "Не удалось экспортировать файл", str(exc))

        def check_external(row: Mapping[str, object]) -> None:
            from PyQt6.QtWidgets import QMessageBox

            version = row.get("_version")
            if version is None or version.external is None:
                QMessageBox.information(
                    self.shell,
                    "Управляемый файл",
                    "Эта версия хранится в базе и не может измениться снаружи.",
                )
                return
            try:
                changed = self.service.external_artifact_changed(version)
            except Exception as exc:
                QMessageBox.warning(self.shell, "Не удалось проверить файл", str(exc))
                return
            QMessageBox.information(
                self.shell,
                "Проверка внешнего файла",
                "Файл изменён или недоступен." if changed else "Файл не изменился.",
            )
        properties = (
            ("Координаты", f"X={x}, Y={y}"),
            ("Номер кадра", (y - 1) * (project.width if project is not None else 1) + x),
            ("Идентификатор кадра", frame_id),
            ("Проект", None if project is None else project.name),
            ("Слой", None if layer is None else layer.name),
            ("Статус", cell.status),
            ("Исполнитель", cell.performer_initials or "—"),
            ("Качество", payload.get("quality")),
            ("Проверка", payload.get("review_status")),
            ("Изображение", None if image is None else image.name),
            ("Версия изображения", payload.get("asset_revision")),
            ("SHA-256 изображения", payload.get("asset_sha256")),
            ("CIF", None if vector is None else vector.name),
            ("Активная версия CIF", payload.get("artifact_version_id")),
            ("Изменён", payload.get("modified_at")),
            (
                "Отсутствующие данные",
                "CIF или изображение отсутствует" if payload.get("missing") else "Нет",
            ),
        )

        def temporal(entry: ObjectHistoryEntry):
            from datetime import datetime

            moment = datetime.fromisoformat(entry.recorded_at)
            historical_project = self.service.get_project(
                project_id,
                as_of=moment,
            )
            if historical_project is None or layer is None:
                return None
            historical_cells = []
            for representation in (image, vector):
                if representation is None:
                    continue
                historical_cells.extend(
                    candidate
                    for candidate in self.service.frame_cells(
                        project_id,
                        layer.id,
                        representation.id,
                        as_of=moment,
                    )
                    if str(candidate.frame_id) == frame_id
                )
            historical = historical_cells[-1] if historical_cells else None
            return ObjectPropertiesSnapshot(
                title=f"Кадр ({x}, {y}) — на {entry.recorded_at}",
                object_kind="frame-temporal",
                properties=(
                    ("Координаты", f"X={x}, Y={y}"),
                    ("Идентификатор кадра", frame_id),
                    ("Проект", historical_project.name),
                    ("Слой", layer.name),
                    (
                        "Статус",
                        "Нет данных" if historical is None else historical.status,
                    ),
                    (
                        "SHA-256",
                        None if historical is None else historical.sha256,
                    ),
                    (
                        "Версия",
                        None
                        if historical is None
                        else historical.artifact_version_id,
                    ),
                ),
                history=tuple(
                    history
                    for history in _history_entries(events)
                    if history.recorded_at <= entry.recorded_at
                ),
            )

        self._open_properties(
            ObjectPropertiesSnapshot(
                title=f"Кадр ({x}, {y})",
                object_kind="frame",
                properties=properties,
                history=_history_entries(events),
                temporal_loader=temporal,
                notes=tuple(
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
                files=tuple(files),
                versions=tuple(versions),
                actions=(
                    (
                        ("Добавить заметку", add_note),
                        ("Изменить последнюю заметку", revise_latest_note),
                    )
                    if Permission.ADD_NOTE in permissions
                    else ()
                ),
                file_actions=(
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
                ),
                version_actions=(
                    (
                        (("Активировать", activate_version),)
                        if Permission.IMPORT_ARTIFACT in permissions
                        else ()
                    )
                    + (
                        ("Экспортировать / открыть", export_or_open_version),
                        ("Проверить внешний файл", check_external),
                    )
                ),
            )
        )
