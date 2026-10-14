"""manager_app: show frame file properties."""
from __future__ import annotations
from collections.abc import Mapping
from kraken_manager.domain.artifacts import ArtifactSeries, ArtifactVersion
from kraken_manager.presentation.qt import (
    ObjectPropertiesSnapshot,
)

from kraken_hub.manager_app_parts.event_matches_node import (
    _event_payload,
    _history_entries,
)


class ShowFrameFilePropertiesMixin:
    def _show_frame_file_properties(
        self,
        label: str,
        series: ArtifactSeries,
        active: ArtifactVersion,
    ) -> None:
        project_id = series.project_id
        principals = {
            str(principal.id): principal.display_name
            for principal in self.service.list_principals(include_inactive=True)
        }
        versions = self.service.artifact_versions(project_id, series.id)
        history = tuple(
            event
            for event in self.service.history(project_id)
            if str(series.id) in str(_event_payload(event))
        )

        def export_version(row: Mapping[str, object]) -> None:
            from PyQt6.QtWidgets import QFileDialog, QMessageBox

            version = row.get("_version")
            if not isinstance(version, ArtifactVersion):
                return
            destination, _selected_filter = QFileDialog.getSaveFileName(
                self.shell,
                "Выгрузить файл",
                version.filename,
            )
            if not destination:
                return
            try:
                self.service.export_artifact_version(
                    project_id,
                    version,
                    destination,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось выгрузить файл",
                    str(exc),
                )

        version_rows = tuple(
            {
                "_version": version,
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
                "active": version.id == active.id,
            }
            for version in versions
        )
        self._open_properties(
            ObjectPropertiesSnapshot(
                title=f"{label}: {active.filename}",
                object_kind="artifact-version",
                properties=(
                    ("Логическое название", series.name),
                    ("Идентификатор серии", str(series.id)),
                    ("Область", series.scope.value),
                    ("Имя файла", active.filename),
                    ("MIME-тип", active.media_type),
                    ("Размер", active.size_bytes),
                    ("SHA-256", active.sha256),
                    (
                        "Хранение",
                        "В базе" if active.blob is not None else "Внешняя ссылка",
                    ),
                    (
                        "Внешний путь",
                        None if active.external is None else active.external.uri,
                    ),
                    ("Автор", principals.get(str(active.author_principal_id))),
                    ("Создан", active.created_at.isoformat()),
                    (
                        "Родительская версия",
                        None
                        if active.parent_version_id is None
                        else str(active.parent_version_id),
                    ),
                    ("Инструмент", active.tool_name),
                    ("Версия инструмента", active.tool_version),
                    ("Provenance", dict(active.parameters)),
                ),
                history=_history_entries(history),
                versions=version_rows,
                version_actions=(("Выгрузить", export_version),),
            )
        )
