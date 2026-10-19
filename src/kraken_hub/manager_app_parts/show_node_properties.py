"""manager_app: show node properties."""
from __future__ import annotations
from collections.abc import Mapping
from kraken_manager.presentation.qt import (
    ObjectPropertiesSnapshot,
    PipelineNode,
)


from kraken_hub.manager_app_parts.event_matches_node import (
    _count_regular_files,
    _creator,
    _event_matches_node,
    _event_payload,
    _history_entries,
    _nested_mapping,
    _plain_value,
)


def _visible_parameters(value: object) -> object:
    if isinstance(value, Mapping):
        hidden = {"format", "version", "input_media_types", "execution_host", "input_transport", "model_version"}
        return {
            key: _visible_parameters(item) for key, item in value.items()
            if not any(part in str(key).lower() for part in ("sha256", "fingerprint", "token")) and key not in hidden
        }
    if isinstance(value, (list, tuple)):
        return [_visible_parameters(item) for item in value]
    return value


class ShowNodePropertiesMixin:
    def _show_node_properties(self, layer_id: str, node: PipelineNode) -> None:
        project_history = tuple(self.service.history(self._project_id))
        snapshot = self._pipeline_snapshot(self._project_id, layer_id)
        if node.kind == "blackbox":
            lane_id = str(node.node_id).removesuffix(":blackbox")
            lane = next((item for item in snapshot.lanes if item.lane_id == lane_id), None)
            hidden = (
                ()
                if lane is None
                else tuple(
                    item
                    for item in lane.nodes
                    if item.kind not in {"source", "vector", "missing"}
                )
            )
            hidden_events = tuple(
                event
                for event in project_history
                if any(_event_matches_node(event, item) for item in hidden)
            )
            self._open_properties(
                ObjectPropertiesSnapshot(
                    title=node.title,
                    object_kind=node.kind,
                    properties=(
                        ("Название", node.title),
                        ("Тип объекта", "Сгруппированные этапы"),
                        ("Идентификатор", node.node_id),
                        ("Путь", None),
                        ("Примечание", node.subtitle),
                        ("Количество файлов", None),
                        ("Кто добавил", None),
                        ("Когда добавлен", None),
                        ("Количество скрытых этапов", len(hidden)),
                        (
                            "Скрытые этапы",
                            [
                                {
                                    "id": item.node_id,
                                    "name": item.title,
                                    "kind": item.kind,
                                    "state": item.state,
                                }
                                for item in hidden
                            ],
                        ),
                    ),
                    history=_history_entries(hidden_events),
                )
            )
            return

        events = tuple(
            event for event in project_history if _event_matches_node(event, node)
        )
        actor, created_at = _creator(events)
        representations = self.service.list_representations(self._project_id, layer_id)
        representation = next(
            (
                item
                for item in representations
                if str(item.id) in {str(node.representation_id), str(node.node_id)}
            ),
            None,
        )
        run_id = str(node.details.get("run_id", ""))
        if not run_id:
            for prefix in ("workspace-job:", "workspace-output:"):
                if str(node.node_id).startswith(prefix):
                    run_id = str(node.node_id)[len(prefix):]
                    break
        derived_run = next(
            (
                item
                for item in self.service.list_derived_runs(self._project_id, layer_id)
                if item.run_id == run_id
            ),
            None,
        )
        kind_labels = {
            "source": "Исходное представление",
            "binary": "Бинарное представление",
            "vector": "Векторное представление",
            "dataset": "Выборка",
            "model": "Модель",
            "job": "Задание",
            "karakal": "Публикация Karakal",
            "confidence": "Уверенность модели",
            "stitching": "Сшивка слоя",
            "missing": "Отсутствующий результат",
        }
        path: object = None
        note: object = None
        file_count: object = None
        properties: list[tuple[str, object]] = [
            ("Название", node.title),
            ("Тип объекта", kind_labels.get(node.kind, node.kind)),
            ("Идентификатор", node.node_id),
            ("Идентификатор слоя", str(layer_id)),
            ("Состояние", node.state or node.subtitle),
        ]
        if node.kind == "stitching":
            properties.extend((("Основной файл", node.details.get("файл")), ("Слой сшит", node.details.get("сшит"))))
        if representation is not None:
            path = representation.source
            note = representation.note
            file_count = self._representation_file_count(layer_id, representation)
            properties.extend(
                (
                    ("Идентификатор представления", str(representation.id)),
                    ("Идентификатор проекта", str(representation.project_id)),
                    ("Вид представления", representation.kind.value),
                    ("Назначение", representation.purpose.value),
                    (
                        "Исходное представление",
                        None
                        if representation.source_image_representation_id is None
                        else str(representation.source_image_representation_id),
                    ),
                    ("Активно", representation.active),
                    ("Состояние объекта", representation.state.value),
                    ("Ревизия", representation.revision),
                    ("Дата объекта", representation.created_at.isoformat()),
                )
            )
        elif derived_run is not None:
            path = derived_run.path
            note = (
                derived_run.provenance.get("notes")
                or derived_run.provenance.get("note")
            )
            file_count = _count_regular_files((derived_run.path,))
            properties.extend(
                (
                    ("Идентификатор запуска", derived_run.run_id),
                    ("Тип результата", derived_run.kind.value),
                    ("Состояние запуска", derived_run.state.value),
                    ("Плагин", derived_run.plugin_id),
                    ("Операция", node.title),
                    ("Дата запуска", derived_run.created_at),
                    ("Параметры", derived_run.provenance.get("parameters", {})),
                )
            )
        else:
            creation = next(
                (
                    event
                    for event in reversed(events)
                    if str(getattr(event, "event_type", ""))
                    in {
                        "PluginJobCreated",
                        "LayerPipelineActionRequested",
                        "KarakalAnalysisPublished",
                    }
                ),
                None,
            )
            latest = events[-1] if events else creation
            creation_payload = {} if creation is None else _event_payload(creation)
            latest_payload = {} if latest is None else _event_payload(latest)
            manifest = _nested_mapping(creation_payload, "manifest")
            job = _nested_mapping(latest_payload, "job")
            if node.kind == "karakal":
                properties.extend(
                    (
                        ("Идентификатор публикации", creation_payload.get("run_id")),
                        (
                            "Номер публикации",
                            creation_payload.get("publication_sequence"),
                        ),
                        (
                            "Версия плагина",
                            creation_payload.get("plugin_version")
                            or node.details.get("версия"),
                        ),
                        (
                            "Параметры",
                            creation_payload.get("parameters")
                            or node.details.get("параметры"),
                        ),
                        (
                            "Отчёт",
                            creation_payload.get("report")
                            or node.details.get("отчёт"),
                        ),
                    )
                )
            elif manifest or job:
                properties.extend(
                    (
                        (
                            "Идентификатор задания",
                            creation_payload.get("plugin_job_id")
                            or job.get("id")
                            or node.details.get("job_id"),
                        ),
                        ("Плагин", manifest.get("plugin_name") or node.details.get("программа")),
                        ("Параметры", node.details.get("параметры") or manifest.get("parameters")),
                        ("Прогресс", job.get("progress") or node.details.get("прогресс")),
                        ("Ошибка", job.get("error") or node.details.get("ошибка")),
                    )
                )
            elif creation is not None:
                properties.extend(
                    (
                        ("Событие создания", getattr(creation, "event_type", "")),
                        ("Плагин", creation_payload.get("plugin_id")),
                        ("Режим", creation_payload.get("mode")),
                        ("Параметры", creation_payload.get("parameters")),
                        ("Параметры события", creation_payload),
                    )
                )
        properties.extend(
            (
                ("Путь", path),
                ("Примечание", note),
                ("Количество файлов", file_count),
                ("Кто добавил", actor),
                ("Когда добавлен", created_at),
                ("Дополнительные свойства", {
                    key: value for key, value in node.details.items()
                    if key in {"время", "автор", "программа", "прогресс", "ошибка", "версия"}
                }),
            )
        )
        parameter_payload = node.details.get("параметры")
        actions = ()
        if parameter_payload:
            def export_parameters() -> None:
                import json
                from pathlib import Path
                from PyQt6.QtWidgets import QFileDialog

                destination, _ = QFileDialog.getSaveFileName(self.shell, "Сохранить параметры", "parameters.json", "JSON (*.json)")
                if destination:
                    try:
                        Path(destination).write_text(json.dumps(_plain_value(parameter_payload), ensure_ascii=False, indent=2), encoding="utf-8")
                    except (OSError, TypeError, ValueError) as exc:
                        self._error(f"Не удалось сохранить параметры: {exc}")

            actions = (("Сохранить параметры в JSON…", export_parameters),)
        self._open_properties(
            ObjectPropertiesSnapshot(
                title=node.title,
                object_kind=node.kind,
                properties=tuple((name, _visible_parameters(value) if name == "Параметры" else value) for name, value in properties),
                history=_history_entries(events),
                actions=actions,
            )
        )
