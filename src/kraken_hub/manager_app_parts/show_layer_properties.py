"""manager_app: show layer properties."""
from __future__ import annotations
from kraken_manager.presentation.qt import (
    ObjectHistoryEntry,
    ObjectPropertiesSnapshot,
)

from kraken_hub.manager_app_parts.event_matches_node import (
    _count_regular_files,
    _creator,
    _event_belongs_to_layer,
    _event_payload,
    _history_entries,
    _plain_value,
)


class ShowLayerPropertiesMixin:
    def _show_layer_properties(self, layer_id: str) -> None:
        layer = next(
            (
                item
                for item in self.service.list_layers(self._project_id)
                if str(item.id) == str(layer_id)
            ),
            None,
        )
        if layer is None:
            self._error("Слой больше не доступен")
            return
        binding = self.service.layer_file_binding(self._project_id, layer_id)
        events = tuple(
            event
            for event in self.service.history(self._project_id)
            if _event_belongs_to_layer(event, layer_id)
        )
        created_events = tuple(
            event
            for event in events
            if str(getattr(event, "event_type", "")) == "LayerCreated"
            and str(_event_payload(event).get("layer_id", "")) == str(layer_id)
        )
        actor, created_at = _creator(created_events or events)
        paths = (
            ()
            if binding is None
            else (
                binding.image_directory,
                binding.ssc_directory,
                binding.prv_directory,
                binding.aux_directory,
            )
        )
        properties: list[tuple[str, object]] = [
            ("Название", layer.name),
            ("Тип объекта", "Слой"),
            ("Идентификатор", str(layer.id)),
            ("Идентификатор проекта", str(layer.project_id)),
            ("Тип слоя", layer.type.value),
            ("Порядок", layer.order),
            ("Состояние", layer.state.value),
            ("Ревизия", layer.revision),
            (
                "Путь",
                None if binding is None else binding.image_directory,
            ),
            ("Примечание", None),
            ("Количество файлов", _count_regular_files(paths)),
            ("Кто добавил", actor),
            ("Когда добавлен", created_at if created_at != "—" else layer.created_at.isoformat()),
        ]
        if binding is not None:
            properties.extend(
                (
                    ("Режим хранения", binding.mode.value),
                    ("Каталог изображений", binding.image_directory),
                    ("Каталог SSC", binding.ssc_directory),
                    ("Каталог PRV", binding.prv_directory),
                    ("Вспомогательный каталог", binding.aux_directory),
                    ("Корень импорта", binding.import_root),
                    ("Известно кадров", len(binding.frame_positions)),
                    ("Позиции кадров", binding.frame_positions),
                    ("Параметры преобразования", _plain_value(binding.conversion)),
                )
            )
        content = self._scope_content(
            self._project_id,
            layer_id=layer.id,
        )

        def temporal(entry: ObjectHistoryEntry):
            from datetime import datetime

            moment = datetime.fromisoformat(entry.recorded_at)
            value = next(
                (
                    candidate
                    for candidate in self.service.list_layers(
                        self._project_id,
                        as_of=moment,
                        include_archived=True,
                    )
                    if candidate.id == layer.id
                ),
                None,
            )
            if value is None:
                return None
            return ObjectPropertiesSnapshot(
                title=f"{value.name} — на {entry.recorded_at}",
                object_kind="layer-temporal",
                properties=(
                    ("Название", value.name),
                    ("Идентификатор", str(value.id)),
                    ("Тип слоя", value.type.value),
                    ("Порядок", value.order),
                    ("Состояние", value.state.value),
                    ("Ревизия", value.revision),
                    ("Создан", value.created_at.isoformat()),
                ),
                history=tuple(
                    history
                    for history in _history_entries(events)
                    if history.recorded_at <= entry.recorded_at
                ),
            )

        self._open_properties(
            ObjectPropertiesSnapshot(
                title=layer.name,
                object_kind="layer",
                properties=tuple(properties),
                history=_history_entries(events),
                temporal_loader=temporal,
                **content,
            )
        )
