"""manager_app: show project properties."""
from __future__ import annotations
from kraken_manager.presentation.qt import (
    ObjectHistoryEntry,
    ObjectPropertiesSnapshot,
)
from kraken_manager.presentation.qt.models import ProjectListItem

from kraken_hub.presentation.qt.parts.event_matches_node import _history_entries


class ShowProjectPropertiesMixin:
    def show_project_properties(self, item: ProjectListItem | None) -> None:
        if item is None:
            return
        project = self.service.get_project(item.project_id)
        if project is None:
            self._error("Проект больше не доступен")
            return
        events = tuple(self.service.history(project.id))
        content = self._scope_content(project.id)

        def properties_for(value) -> tuple[tuple[str, object], ...]:
            return (
                ("Название", value.name),
                ("Идентификатор", str(value.id)),
                ("Размер матрицы", f"{value.width} × {value.height}"),
                ("Ориентация", value.orientation.value),
                ("Профиль хранения", value.storage_profile),
                ("Состояние", value.state.value),
                ("Ревизия", value.revision),
                ("Создан", value.created_at.isoformat()),
            )

        def temporal(entry: ObjectHistoryEntry):
            from datetime import datetime

            moment = datetime.fromisoformat(entry.recorded_at)
            value = self.service.get_project(project.id, as_of=moment)
            if value is None:
                return None
            return ObjectPropertiesSnapshot(
                title=f"{value.name} — на {entry.recorded_at}",
                object_kind="project-temporal",
                properties=properties_for(value),
                history=tuple(
                    history
                    for history in _history_entries(events)
                    if history.recorded_at <= entry.recorded_at
                ),
            )

        self._open_properties(
            ObjectPropertiesSnapshot(
                title=project.name,
                object_kind="project",
                properties=properties_for(project),
                history=_history_entries(events),
                temporal_loader=temporal,
                **content,
            )
        )
