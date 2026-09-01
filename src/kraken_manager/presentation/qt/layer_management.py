"""Modeless layer manager and provenance graph for a project workspace."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from PyQt6.QtCore import QLineF, QPointF, QRectF, QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QHeaderView,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .models import LayerListItem


@dataclass(frozen=True, slots=True)
class PipelineNode:
    node_id: str
    title: str
    kind: str
    subtitle: str = ""
    representation_id: str = ""
    active: bool = False
    state: str = ""
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PipelineLane:
    lane_id: str
    title: str
    nodes: tuple[PipelineNode, ...]
    edges: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class LayerPipelineSnapshot:
    project_id: str
    layer_id: str
    lanes: tuple[PipelineLane, ...] = ()


@dataclass(frozen=True, slots=True)
class ObjectHistoryEntry:
    recorded_at: str
    actor: str
    event_type: str
    payload: object = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ObjectPropertiesSnapshot:
    title: str
    object_kind: str
    properties: tuple[tuple[str, object], ...]
    history: tuple[ObjectHistoryEntry, ...] = ()
    notes: tuple[Mapping[str, object], ...] = ()
    files: tuple[Mapping[str, object], ...] = ()
    versions: tuple[Mapping[str, object], ...] = ()
    actions: tuple[tuple[str, Callable[[], None]], ...] = field(
        default=(),
        compare=False,
        repr=False,
    )
    file_actions: tuple[
        tuple[str, Callable[[Mapping[str, object]], None]], ...
    ] = field(default=(), compare=False, repr=False)
    version_actions: tuple[
        tuple[str, Callable[[Mapping[str, object]], None]], ...
    ] = field(default=(), compare=False, repr=False)
    temporal_loader: Callable[
        [ObjectHistoryEntry],
        ObjectPropertiesSnapshot | None,
    ] | None = field(default=None, compare=False, repr=False)


def _display_value(value: object) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, (Mapping, list, tuple, set)):
        serializable = dict(value) if isinstance(value, Mapping) else value
        return json.dumps(
            list(value) if isinstance(value, set) else serializable,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    if isinstance(value, bool):
        return "Да" if value else "Нет"
    return str(value)


def _local_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return str(value) or "—"
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    return parsed.strftime("%d.%m.%Y %H:%M:%S")


_EVENT_TYPE_LABELS: dict[str, str] = {
    "ProjectCreated": "Создан проект",
    "ProjectRenamed": "Переименован проект",
    "ProjectArchived": "Проект архивирован",
    "ProjectRestored": "Проект восстановлен",
    "ProjectRoleAssigned": "Назначена роль",
    "ProjectRoleRevoked": "Отозвана роль",
    "LayerCreated": "Создан слой",
    "LayerRenamed": "Переименован слой",
    "LayerReordered": "Изменён порядок слоя",
    "LayersReordered": "Изменён порядок слоёв",
    "LayerArchived": "Слой архивирован",
    "RepresentationCreated": "Создана репрезентация",
    "RepresentationRenamed": "Переименована репрезентация",
    "RepresentationNoteUpdated": "Обновлено примечание репрезентации",
    "RepresentationActivated": "Репрезентация активирована",
    "RepresentationDeactivated": "Репрезентация деактивирована",
    "RepresentationArchived": "Репрезентация архивирована",
    "ArtifactSeriesCreated": "Добавлен файл",
    "ArtifactSeriesRenamed": "Файл переименован",
    "ArtifactSeriesArchived": "Файл перенесён в архив",
    "ArtifactVersionCreated": "Создана версия файла",
    "ArtifactVersionActivated": "Выбрана версия файла",
    "ExternalArtifactVersionAdded": "Добавлена версия внешнего файла",
    "NoteCreated": "Добавлена заметка",
    "NoteRevised": "Изменена заметка",
    "ReviewBatchCreated": "Создано задание на проверку",
    "ReviewBatchIssued": "Задание выдано",
    "ReviewBatchReexported": "Пакет выдан повторно",
    "ReviewReturnCommitted": "Результат проверки загружен",
    "ReviewBatchAccepted": "Результат проверки принят",
    "ReviewChangesRequested": "Запрошена доработка",
    "ReviewBatchCancelled": "Проверка отменена",
    "PluginJobCreated": "Создано задание плагина",
    "PluginResultAwaitingAuthorization": "Результат плагина ожидает подтверждения",
    "PluginPartialResultReceived": "Получен частичный результат плагина",
    "PluginResultImported": "Результат плагина импортирован",
    "PluginJobFailed": "Задание плагина завершилось ошибкой",
    "PluginJobCancelled": "Задание плагина отменено",
    "PluginJobRetried": "Задание плагина запущено повторно",
    "PluginJobSynchronized": "Состояние задания плагина синхронизировано",
    "LayerPipelineActionRequested": "Запущено действие конвейера",
    "LayerPipelineActionRemoved": "Удалено действие конвейера",
    "KarakalAnalysisPublished": "Опубликован анализ Karakal",
}

_FIELD_LABELS: dict[str, str] = {
    "name": "Название",
    "width": "Ширина",
    "height": "Высота",
    "orientation": "Ориентация",
    "storage_profile": "Профиль хранения",
    "state": "Состояние",
    "type": "Тип",
    "kind": "Вид",
    "purpose": "Назначение",
    "note": "Примечание",
    "source": "Источник",
    "active": "Активна",
    "order": "Порядок",
    "role": "Роль",
    "instructions": "Инструкции",
    "reason": "Причина",
    "filename": "Файл",
    "media_type": "Тип файла",
    "size_bytes": "Размер",
    "sha256": "SHA-256",
    "capability": "Capability",
    "progress": "Прогресс",
    "error": "Ошибка",
    "plugin_id": "Плагин",
    "plugin_version": "Версия плагина",
    "action": "Действие",
    "mode": "Режим",
    "agent_state": "Состояние агента",
    "scope": "Область",
    "body": "Текст",
    "revision": "Ревизия",
    "batch_revision": "Ревизия задания",
    "file_count": "Файлов",
    "total_size_bytes": "Общий размер",
    "publication_sequence": "Номер публикации",
    "partial": "Частичный результат",
    "due_at": "Срок",
    "created_at": "Создано",
    "updated_at": "Обновлено",
    "finished_at": "Завершено",
    "recorded_at": "Записано",
}

_STATE_LABELS: dict[str, str] = {
    "active": "активен",
    "archived": "архивирован",
    "draft": "черновик",
    "issued": "выдано",
    "awaiting_acceptance": "ожидает приёмки",
    "changes_requested": "запрошена доработка",
    "completed": "завершено",
    "cancelled": "отменено",
    "queued": "в очереди",
    "running": "выполняется",
    "succeeded": "успешно",
    "failed": "ошибка",
    "cancelled_by_user": "отменено пользователем",
    "launched": "запущено",
}

_ROLE_LABELS: dict[str, str] = {
    "owner": "владелец",
    "editor": "редактор",
    "viewer": "наблюдатель",
    "reviewer": "проверяющий",
}

_ID_FIELD_SUFFIXES = (
    "_id",
    "_ids",
    "principal_id",
    "performer_id",
    "assignee_id",
    "created_by",
    "assigned_by",
    "author_principal_id",
    "actor_principal_id",
    "target_representation_id",
    "source_image_representation_id",
    "plugin_job_id",
    "review_batch_id",
    "artifact_series_id",
    "artifact_version_id",
    "candidate_version_ids",
    "deactivated_representation_ids",
    "action_event_id",
    "package_id",
    "run_id",
    "node_id",
    "frame_id",
    "series_id",
    "layer_id",
    "project_id",
    "representation_id",
    "note_id",
    "manifest_fingerprint",
    "request_fingerprint",
    "return_fingerprint",
)


def _event_type_label(event_type: str) -> str:
    label = _EVENT_TYPE_LABELS.get(event_type)
    if label:
        return label
    spaced = "".join(f" {char}" if char.isupper() else char for char in event_type).strip()
    return spaced or event_type or "—"


def _is_uuid_like(value: object) -> bool:
    text = str(value or "").strip()
    if len(text) != 36:
        return False
    parts = text.split("-")
    return len(parts) == 5 and all(
        part and all(character in "0123456789abcdefABCDEF" for character in part)
        for part in parts
    )


def _format_bytes(value: object) -> str | None:
    try:
        size = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if size < 0:
        return None
    units = ("Б", "КБ", "МБ", "ГБ", "ТБ")
    amount = float(size)
    unit = units[0]
    for candidate in units:
        unit = candidate
        if amount < 1024 or candidate == units[-1]:
            break
        amount /= 1024
    if unit == "Б":
        return f"{size} {unit}"
    return f"{amount:.1f} {unit}"


def _format_scalar(key: str, value: object) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "Да" if value else "Нет"
    if key.endswith("_at") or key in {"due_at", "created_at", "updated_at", "finished_at", "recorded_at"}:
        return _local_timestamp(str(value))
    if key in {"size_bytes", "total_size_bytes"}:
        formatted = _format_bytes(value)
        if formatted is not None:
            return formatted
    if key == "progress":
        try:
            return f"{float(value) * 100:.0f}%"
        except (TypeError, ValueError):
            pass
    if key == "role":
        return _ROLE_LABELS.get(str(value), str(value))
    if key == "state" or key.endswith("_state"):
        return _STATE_LABELS.get(str(value), str(value))
    if key in {"width", "height", "order", "revision", "batch_revision", "file_count", "publication_sequence"}:
        return str(value)
    if key == "sha256":
        text = str(value)
        return text if len(text) <= 16 else f"{text[:12]}…"
    return str(value)


def _field_label(key: str) -> str:
    return _FIELD_LABELS.get(key, key.replace("_", " "))


def _should_skip_key(key: str) -> bool:
    if key in _ID_FIELD_SUFFIXES:
        return True
    if key.endswith("_id") or key.endswith("_ids"):
        return True
    if key.endswith("_fingerprint"):
        return True
    return key in {
        "blob",
        "external",
        "comparisons",
        "frame_confidence",
        "report",
        "parameters",
        "manifest",
        "result",
        "items",
        "selection",
        "deactivated",
    }


def _append_lines(lines: list[str], key: str, value: object) -> None:
    if isinstance(value, (Mapping, list, tuple, set)):
        return
    text = _format_scalar(key, value)
    if text and text != "—":
        lines.append(f"{_field_label(key)}: {text}")


def _append_mapping_fields(lines: list[str], payload: Mapping[str, object], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, Mapping):
            continue
        if _is_uuid_like(value):
            continue
        _append_lines(lines, key, value)


def _selection_summary(selection: object) -> str | None:
    if not isinstance(selection, Mapping):
        return None
    mode = str(selection.get("mode") or selection.get("kind") or "").strip()
    frames = selection.get("frames") or selection.get("frame_ids") or selection.get("items")
    count: int | None = None
    if isinstance(frames, (list, tuple, set)):
        count = len(frames)
    elif "count" in selection:
        try:
            count = int(selection["count"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            count = None
    parts: list[str] = []
    if mode:
        parts.append(mode)
    if count is not None:
        parts.append(f"{count} кадр(ов)")
    return ", ".join(parts) if parts else None


def _format_history_payload(event_type: str, payload: object) -> str:
    if not isinstance(payload, Mapping) or not payload:
        return "—"

    lines: list[str] = []
    data = dict(payload)

    nested_keys = (
        ("project", ("name", "width", "height", "orientation", "storage_profile", "state")),
        ("layer", ("name", "type", "order", "state")),
        ("representation", ("name", "kind", "purpose", "note", "source", "active", "state")),
        ("job", ("capability", "state", "progress", "error")),
    )
    for nested_key, fields in nested_keys:
        nested = data.get(nested_key)
        if isinstance(nested, Mapping):
            _append_mapping_fields(lines, nested, fields)
            selection = _selection_summary(nested.get("selection"))
            if selection:
                lines.append(f"Выборка: {selection}")

    _append_mapping_fields(
        lines,
        data,
        (
            "name",
            "width",
            "height",
            "orientation",
            "storage_profile",
            "state",
            "type",
            "kind",
            "purpose",
            "note",
            "source",
            "active",
            "order",
            "role",
            "instructions",
            "reason",
            "filename",
            "media_type",
            "size_bytes",
            "sha256",
            "capability",
            "progress",
            "error",
            "plugin_id",
            "plugin_version",
            "action",
            "mode",
            "agent_state",
            "scope",
            "body",
            "revision",
            "batch_revision",
            "file_count",
            "total_size_bytes",
            "publication_sequence",
            "partial",
            "due_at",
            "created_at",
            "updated_at",
            "finished_at",
        ),
    )

    selection = _selection_summary(data.get("selection"))
    if selection:
        lines.append(f"Выборка: {selection}")

    items = data.get("items")
    if isinstance(items, (list, tuple)):
        lines.append(f"Элементов: {len(items)}")

    deactivated = data.get("deactivated") or data.get("deactivated_representation_ids")
    if isinstance(deactivated, (list, tuple)) and deactivated:
        names = [
            str(item.get("name"))
            for item in deactivated
            if isinstance(item, Mapping) and item.get("name")
        ]
        if names:
            lines.append(f"Деактивированы: {', '.join(names)}")
        else:
            lines.append(f"Деактивировано: {len(deactivated)}")

    candidates = data.get("candidate_version_ids")
    if isinstance(candidates, (list, tuple)) and candidates:
        lines.append(f"Файлов на подтверждение: {len(candidates)}")

    artifact_ids = data.get("artifact_version_ids")
    if isinstance(artifact_ids, (list, tuple)) and artifact_ids:
        lines.append(f"Сохранённых версий файлов: {len(artifact_ids)}")

    parameters = data.get("parameters")
    if isinstance(parameters, Mapping) and parameters:
        preview = ", ".join(
            f"{key}={value}"
            for key, value in list(parameters.items())[:4]
            if not isinstance(value, (Mapping, list, tuple, set))
        )
        if preview:
            lines.append(f"Параметры: {preview}")

    if not lines:
        for key, value in data.items():
            if _should_skip_key(key) or isinstance(value, (Mapping, list, tuple, set)):
                continue
            if _is_uuid_like(value):
                continue
            _append_lines(lines, key, value)
            if len(lines) >= 6:
                break

    if not lines:
        return "Подробности недоступны"

    # Deduplicate while preserving order.
    unique: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return "\n".join(unique)


class ObjectPropertiesDialog(QDialog):
    """Read-only object metadata and event history."""

    def __init__(
        self,
        snapshot: ObjectPropertiesSnapshot,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.snapshot = snapshot
        self.setObjectName("objectPropertiesDialog")
        self.setWindowTitle(f"Свойства: {snapshot.title}")
        self.setModal(True)
        self.resize(880, 620)

        tabs = QTabWidget(self)
        tabs.setObjectName("objectPropertiesTabs")
        self.properties_table = self._table(("Свойство", "Значение"))
        self.properties_table.setObjectName("objectPropertiesTable")
        self.properties_table.setRowCount(len(snapshot.properties))
        for row, (name, value) in enumerate(snapshot.properties):
            self.properties_table.setItem(row, 0, QTableWidgetItem(str(name)))
            value_item = QTableWidgetItem(_display_value(value))
            value_item.setToolTip(value_item.text())
            self.properties_table.setItem(row, 1, value_item)
        self.properties_table.resizeRowsToContents()
        tabs.addTab(self.properties_table, "Свойства")

        self.history_table = self._table(
            ("Дата и время", "Пользователь", "Событие", "Подробности")
        )
        self.history_table.setObjectName("objectHistoryTable")
        self.history_table.setRowCount(len(snapshot.history))
        for row, entry in enumerate(snapshot.history):
            timestamp = QTableWidgetItem(_local_timestamp(entry.recorded_at))
            timestamp.setToolTip(entry.recorded_at)
            self.history_table.setItem(row, 0, timestamp)
            self.history_table.setItem(row, 1, QTableWidgetItem(entry.actor or "—"))
            event_item = QTableWidgetItem(_event_type_label(entry.event_type))
            event_item.setToolTip(entry.event_type)
            self.history_table.setItem(row, 2, event_item)
            summary = _format_history_payload(entry.event_type, entry.payload)
            details = QTableWidgetItem(summary)
            details.setToolTip(_display_value(entry.payload))
            self.history_table.setItem(row, 3, details)
        self.history_table.resizeRowsToContents()
        if snapshot.temporal_loader is not None:
            self.history_table.setToolTip(
                "Дважды щёлкните событие, чтобы открыть состояние объекта на этот момент"
            )
            self.history_table.cellDoubleClicked.connect(
                self._open_temporal_state
            )
        tabs.addTab(self.history_table, f"История изменений ({len(snapshot.history)})")
        self.notes_table = self._mapping_table(
            snapshot.notes,
            (
                ("revision", "Ревизия"),
                ("body", "Текст"),
                ("author", "Автор"),
                ("recorded_at", "Дата и время"),
            ),
        )
        self.notes_table.setObjectName("objectNotesTable")
        tabs.addTab(
            self._action_tab(
                self.notes_table,
                snapshot.notes,
                (),
                general_actions=snapshot.actions,
            ),
            f"Заметки ({len(snapshot.notes)})",
        )
        self.files_table = self._mapping_table(
            snapshot.files,
            (
                ("name", "Название"),
                ("scope", "Тип"),
                ("active_version", "Активная версия"),
                ("archived", "Архив"),
            ),
        )
        self.files_table.setObjectName("objectFilesTable")
        tabs.addTab(
            self._action_tab(
                self.files_table,
                snapshot.files,
                snapshot.file_actions,
            ),
            f"Файлы ({len(snapshot.files)})",
        )
        self.versions_table = self._mapping_table(
            snapshot.versions,
            (
                ("filename", "Имя"),
                ("size", "Размер"),
                ("sha256", "SHA-256"),
                ("author", "Автор"),
                ("created_at", "Дата"),
                ("parent", "Родитель"),
                ("tool", "Инструмент"),
                ("provenance", "Provenance"),
                ("active", "Активная"),
            ),
        )
        self.versions_table.setObjectName("objectVersionsTable")
        tabs.addTab(
            self._action_tab(
                self.versions_table,
                snapshot.versions,
                snapshot.version_actions,
            ),
            f"Версии ({len(snapshot.versions)})",
        )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("Закрыть")
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs, 1)
        layout.addWidget(buttons)

    def _open_temporal_state(self, row: int, _column: int) -> None:
        loader = self.snapshot.temporal_loader
        if loader is None or not 0 <= row < len(self.snapshot.history):
            return
        historical = loader(self.snapshot.history[row])
        if historical is None:
            return
        ObjectPropertiesDialog(historical, self).exec()

    def _table(self, headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(headers), self)
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.setWordWrap(True)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(headers) - 1, QHeaderView.ResizeMode.Stretch)
        return table

    def _mapping_table(
        self,
        rows: tuple[Mapping[str, object], ...],
        columns: tuple[tuple[str, str], ...],
    ) -> QTableWidget:
        table = self._table(tuple(label for _key, label in columns))
        table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, (key, _label) in enumerate(columns):
                value = values.get(key)
                if key in {"recorded_at", "created_at"} and value:
                    text = _local_timestamp(str(value))
                else:
                    text = _display_value(value)
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                table.setItem(row, column, item)
        table.resizeRowsToContents()
        return table

    def _action_tab(
        self,
        table: QTableWidget,
        rows: tuple[Mapping[str, object], ...],
        row_actions: tuple[
            tuple[str, Callable[[Mapping[str, object]], None]], ...
        ],
        *,
        general_actions: tuple[tuple[str, Callable[[], None]], ...] = (),
    ) -> QWidget:
        host = QWidget(self)
        layout = QVBoxLayout(host)
        action_layout = QHBoxLayout()
        for label, callback in general_actions:
            button = QPushButton(label, host)
            button.clicked.connect(lambda _checked=False, action=callback: action())
            action_layout.addWidget(button)
        for label, callback in row_actions:
            button = QPushButton(label, host)

            def invoke(
                _checked: bool = False,
                *,
                action=callback,
                source=table,
                values=rows,
            ) -> None:
                row = source.currentRow()
                if 0 <= row < len(values):
                    action(values[row])

            button.clicked.connect(invoke)
            action_layout.addWidget(button)
        action_layout.addStretch(1)
        layout.addLayout(action_layout)
        layout.addWidget(table, 1)
        return host


class LayerOrderList(QListWidget):
    orderChanged = pyqtSignal(object)
    layerSelected = pyqtSignal(str)
    layerActionRequested = pyqtSignal(str, str)
    layerPropertiesRequested = pyqtSignal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        action_availability: Callable[[str], tuple[bool, str]] | None = None,
    ) -> None:
        super().__init__(parent)
        self._action_availability = action_availability or (lambda _action: (True, ""))
        self.setObjectName("layerManagerList")
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.model().rowsMoved.connect(lambda *_: self.orderChanged.emit(self.layer_ids()))
        self.currentItemChanged.connect(self._emit_selection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    def set_layers(self, layers: list[LayerListItem], selected_id: str = "") -> None:
        self.blockSignals(True)
        self.clear()
        selected_row = 0
        for row, layer in enumerate(layers):
            item = QListWidgetItem(layer.name)
            item.setData(Qt.ItemDataRole.UserRole, layer.layer_id)
            item.setToolTip(layer.layer_type)
            item.setForeground(QColor(layer.color))
            self.addItem(item)
            if layer.layer_id == selected_id:
                selected_row = row
        if self.count():
            self.setCurrentRow(selected_row)
        self.blockSignals(False)

    def layer_ids(self) -> tuple[str, ...]:
        return tuple(str(self.item(row).data(Qt.ItemDataRole.UserRole)) for row in range(self.count()))

    def select_layer(self, layer_id: str) -> None:
        for row in range(self.count()):
            if str(self.item(row).data(Qt.ItemDataRole.UserRole)) == str(layer_id):
                self.setCurrentRow(row)
                return

    def _emit_selection(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is not None:
            self.layerSelected.emit(str(current.data(Qt.ItemDataRole.UserRole)))

    def _context_menu(self, position) -> None:
        item = self.itemAt(position)
        if item is None:
            return
        layer_id = str(item.data(Qt.ItemDataRole.UserRole))
        menu = QMenu(self)
        add_images = menu.addAction("Добавить слой изображений…")
        karakal = menu.addAction("Отправить слой в Karakal")
        menu.addSeparator()
        rename_layer = menu.addAction("Переименовать…")
        archive_layer = menu.addAction("Архивировать")
        delete_layer = menu.addAction("Удалить слой…")
        menu.addSeparator()
        properties = menu.addAction("Свойства")
        by_action = {
            add_images: "add_image_representation",
            karakal: "karakal",
            rename_layer: "rename_layer",
            archive_layer: "archive_layer",
            delete_layer: "delete_layer",
        }
        for menu_action, code in by_action.items():
            enabled, reason = self._action_availability(code)
            menu_action.setEnabled(enabled)
            if not enabled and reason.startswith("Недостаточно прав"):
                menu_action.setVisible(False)
            if reason:
                menu_action.setToolTip(reason)
                menu_action.setStatusTip(reason)
        selected = menu.exec(self.viewport().mapToGlobal(position))
        if selected is properties:
            self.layerPropertiesRequested.emit(layer_id)
        elif selected in by_action:
            self.layerActionRequested.emit(layer_id, by_action[selected])


class PipelineScene(QGraphicsScene):
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # type: ignore[override]
        painter.fillRect(rect, QColor("#1b1d1b"))
        minor, major = 20, 100
        for spacing, color in ((minor, "#272a27"), (major, "#343834")):
            pen = QPen(QColor(color), 0)
            painter.setPen(pen)
            left = int(rect.left()) - (int(rect.left()) % spacing)
            top = int(rect.top()) - (int(rect.top()) % spacing)
            for x in range(left, int(rect.right()) + spacing, spacing):
                painter.drawLine(QLineF(float(x), rect.top(), float(x), rect.bottom()))
            for y in range(top, int(rect.bottom()) + spacing, spacing):
                painter.drawLine(QLineF(rect.left(), float(y), rect.right(), float(y)))


class PipelineNodeItem(QGraphicsRectItem):
    COLORS = {
        "source": "#4d738a",
        "binary": "#278b78",
        "vector": "#9a6324",
        "dataset": "#73559b",
        "model": "#9b4055",
        "job": "#4c5350",
        "blackbox": "#303430",
        "missing": "#303430",
        "karakal": "#3d718b",
    }

    def __init__(
        self,
        node: PipelineNode,
        *,
        activate: Callable[[PipelineNode], None],
        request_action: Callable[[PipelineNode, str], None],
        request_properties: Callable[[PipelineNode], None],
        expand: Callable[[], None],
        collapse: Callable[[], None] | None,
        action_availability: Callable[[str], tuple[bool, str]],
        position_changed: Callable[[], None],
    ) -> None:
        super().__init__(0, 0, 190, 72)
        self.node = node
        self._activate = activate
        self._request_action = request_action
        self._request_properties = request_properties
        self._expand = expand
        self._collapse = collapse
        self._action_availability = action_availability
        self._position_changed = position_changed
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(self.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setBrush(QColor(self.COLORS.get(node.kind, "#4c5350")))
        pen = QPen(QColor("#f59e0b" if node.active else "#101210"), 3 if node.active else 1.5)
        pen.setStyle(Qt.PenStyle.DashLine if node.kind == "missing" else Qt.PenStyle.SolidLine)
        self.setPen(pen)
        title = QGraphicsSimpleTextItem(node.title, self)
        title.setBrush(QColor("#f3f4f6"))
        title.setPos(12, 9)
        if node.subtitle:
            subtitle = QGraphicsSimpleTextItem(node.subtitle[:52], self)
            subtitle.setBrush(QColor("#c3c8c3"))
            subtitle.setPos(12, 38)
        self.setToolTip("\n".join(f"{key}: {value}" for key, value in node.details.items()))
        port_color = QColor("#22d3ee" if node.kind in {"source", "binary"} else "#facc15")
        for x in (-5.0, 185.0):
            port = QGraphicsEllipseItem(x, 31.0, 10.0, 10.0, self)
            port.setBrush(port_color)
            port.setPen(QPen(QColor("#111827"), 1.0))

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):  # type: ignore[override]
        result = super().itemChange(change, value)
        if change is QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._position_changed()
        return result

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        if event.button() is Qt.MouseButton.LeftButton and self.node.kind == "blackbox":
            self._expand()
            event.accept()
            return
        super().mousePressEvent(event)
        if event.button() is Qt.MouseButton.LeftButton:
            self._activate(self.node)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event) -> None:  # type: ignore[override]
        actions = {
            "source": (
                ("Подготовить выборку в Contour", "prepare_dataset"),
                ("Распознать готовой моделью", "recognize_external"),
                ("Получить CIF в Contour", "vectorize"),
                (
                    "Удалить слой изображений из проекта",
                    "archive_representation",
                ),
            ),
            "dataset": (("Обучить модель в NeuralImage", "train"),),
            "model": (("Распознать исходники", "recognize"),),
            "binary": (("Получить CIF в Contour", "vectorize"),),
            "vector": (("Добавить CIF из внешнего источника…", "add_external_vector"),),
            "missing": (("Добавить CIF из внешнего источника…", "add_external_vector"),),
        }.get(self.node.kind, ())
        if bool(self.node.details.get("deletable", False)):
            actions = (*actions, ("Удалить шаг из pipeline", "delete_pipeline_step"))
        if self.node.representation_id:
            lifecycle = (
                ("Переименовать…", "rename_representation"),
                ("Изменить заметку…", "edit_representation_note"),
                (
                    "Деактивировать" if self.node.active else "Активировать",
                    "deactivate_representation" if self.node.active else "activate_representation",
                ),
                ("Архивировать", "archive_representation"),
            )
            actions = (*actions, *lifecycle)
        if self._collapse is not None:
            actions = (*actions, ("Свернуть", "collapse_pipeline"))
        menu = QMenu()
        by_action = {}
        for label, code in actions:
            menu_action = menu.addAction(label)
            enabled, reason = (
                (True, "")
                if code == "collapse_pipeline"
                else self._action_availability(code)
            )
            menu_action.setEnabled(enabled)
            if not enabled and reason.startswith("Недостаточно прав"):
                menu_action.setVisible(False)
            if reason:
                menu_action.setToolTip(reason)
                menu_action.setStatusTip(reason)
            by_action[menu_action] = code
        if actions:
            menu.addSeparator()
        properties = menu.addAction("Свойства")
        selected = menu.exec(event.screenPos())
        if selected is properties:
            self._request_properties(self.node)
        elif selected in by_action:
            code = by_action[selected]
            if code == "collapse_pipeline" and self._collapse is not None:
                self._collapse()
            else:
                self._request_action(self.node, code)


class PipelineGraphView(QGraphicsView):
    nodeActivated = pyqtSignal(object)
    nodeActionRequested = pyqtSignal(object, str)
    nodePropertiesRequested = pyqtSignal(object)

    def __init__(
        self,
        settings: QSettings,
        parent: QWidget | None = None,
        *,
        action_availability: Callable[[str], tuple[bool, str]] | None = None,
    ) -> None:
        self.graph_scene = PipelineScene()
        super().__init__(self.graph_scene, parent)
        self.setObjectName("layerPipelineGraph")
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._settings = settings
        self._action_availability = action_availability or (lambda _action: (True, ""))
        self._snapshot: LayerPipelineSnapshot | None = None
        self._expanded_lane_id = ""
        self._items: dict[str, PipelineNodeItem] = {}
        self._edges: list[tuple[QGraphicsPathItem, PipelineNodeItem, PipelineNodeItem]] = []

    def set_snapshot(self, snapshot: LayerPipelineSnapshot) -> None:
        self.save_layout()
        self._snapshot = snapshot
        self._expanded_lane_id = str(
            self._settings.value(self._key("expanded-lane"), "", type=str)
        )
        if self._expanded_lane_id not in {lane.lane_id for lane in snapshot.lanes}:
            self._expanded_lane_id = ""
        self._rebuild()

    def _key(self, suffix: str) -> str:
        snap = self._snapshot
        return f"layer-manager/{snap.project_id}/{snap.layer_id}/{suffix}" if snap else f"layer-manager/{suffix}"

    def _rebuild(self) -> None:
        self.graph_scene.clear()
        self._items.clear()
        self._edges.clear()
        snapshot = self._snapshot
        if snapshot is None:
            return
        y = 30.0
        for lane in snapshot.lanes:
            visible_nodes = list(lane.nodes)
            visible_edges = list(lane.edges)
            internal_nodes = [
                node
                for node in lane.nodes
                if node.kind not in {"source", "vector", "missing"}
            ]
            is_expanded = lane.lane_id == self._expanded_lane_id
            blackbox = PipelineNode(
                f"{lane.lane_id}:blackbox",
                "Чёрный ящик",
                "blackbox",
                "Нажмите, чтобы раскрыть",
            )
            if internal_nodes and not is_expanded:
                sources = [node for node in lane.nodes if node.kind == "source"]
                vectors = [node for node in lane.nodes if node.kind in {"vector", "missing"}]
                visible_nodes = [*sources, blackbox, *vectors]
                visible_edges = []
                if sources:
                    visible_edges.append((sources[0].node_id, blackbox.node_id))
                for vector in vectors:
                    visible_edges.append((blackbox.node_id, vector.node_id))
            for column, node in enumerate(visible_nodes):
                belongs_to_expanded_box = is_expanded and node in internal_nodes
                item = PipelineNodeItem(
                    node,
                    activate=lambda value: self.nodeActivated.emit(value),
                    request_action=lambda value, action: self.nodeActionRequested.emit(value, action),
                    request_properties=lambda value: self.nodePropertiesRequested.emit(value),
                    expand=lambda lane_id=lane.lane_id: self.expand_lane(lane_id),
                    collapse=(
                        (lambda lane_id=lane.lane_id: self.collapse_lane(lane_id))
                        if belongs_to_expanded_box
                        else None
                    ),
                    action_availability=self._action_availability,
                    position_changed=self._update_edges,
                )
                saved = self._settings.value(self._key(f"node/{node.node_id}"))
                if saved is not None and isinstance(saved, QPointF):
                    item.setPos(saved)
                else:
                    item.setPos(35 + column * 245, y)
                self.graph_scene.addItem(item)
                self._items[node.node_id] = item
            for source_id, target_id in visible_edges:
                source, target = self._items.get(source_id), self._items.get(target_id)
                if source is None or target is None:
                    continue
                edge = QGraphicsPathItem()
                edge.setPen(QPen(QColor("#9ca39c"), 2))
                edge.setZValue(-1)
                self.graph_scene.addItem(edge)
                self._edges.append((edge, source, target))
            self._update_edges()
            y += 135
        self.graph_scene.setSceneRect(self.graph_scene.itemsBoundingRect().adjusted(-80, -80, 120, 120))

    def _update_edges(self) -> None:
        for edge, source, target in self._edges:
            source_rect = source.sceneBoundingRect()
            target_rect = target.sceneBoundingRect()
            start = QPointF(source_rect.right(), source_rect.center().y())
            end = QPointF(target_rect.left(), target_rect.center().y())
            bend = max(60.0, abs(end.x() - start.x()) * 0.45)
            path = QPainterPath(start)
            path.cubicTo(start.x() + bend, start.y(), end.x() - bend, end.y(), end.x(), end.y())
            edge.setPath(path)

    def expand_lane(self, lane_id: str) -> None:
        self.save_layout()
        self._expanded_lane_id = str(lane_id)
        self._settings.setValue(self._key("expanded-lane"), self._expanded_lane_id)
        self._rebuild()

    def collapse_lane(self, lane_id: str) -> None:
        if self._expanded_lane_id != str(lane_id):
            return
        self.save_layout()
        self._expanded_lane_id = ""
        self._settings.remove(self._key("expanded-lane"))
        self._rebuild()

    def save_layout(self) -> None:
        for identifier, item in self._items.items():
            self._settings.setValue(self._key(f"node/{identifier}"), item.pos())

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)


class LayerManagerDialog(QDialog):
    addLayerRequested = pyqtSignal()
    layerSelected = pyqtSignal(str)
    orderChanged = pyqtSignal(object)
    representationActivated = pyqtSignal(str)
    nodeActionRequested = pyqtSignal(str, object, str)
    layerActionRequested = pyqtSignal(str, str)
    nodePropertiesRequested = pyqtSignal(str, object)
    layerPropertiesRequested = pyqtSignal(str)

    def __init__(
        self,
        project_id: str,
        parent: QWidget | None = None,
        *,
        action_availability: Callable[[str], tuple[bool, str]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.project_id = str(project_id)
        self.setObjectName("layerManagerDialog")
        self.setWindowTitle("Управление слоями")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.resize(1180, 720)
        self.settings = QSettings("Kraken", "KrakenHub")
        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Orientation.Horizontal)
        layer_panel = QWidget(splitter)
        layer_layout = QVBoxLayout(layer_panel)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_list = LayerOrderList(action_availability=action_availability)
        self.add_layer_button = QPushButton("Добавить слой", layer_panel)
        self.add_layer_button.setObjectName("layerManagerAddLayer")
        self.add_layer_button.setToolTip("Создать слой из папки или подключить внешние каталоги")
        layer_layout.addWidget(self.layer_list, 1)
        layer_layout.addWidget(self.add_layer_button)
        self.graph = PipelineGraphView(self.settings, action_availability=action_availability)
        splitter.addWidget(layer_panel)
        splitter.addWidget(self.graph)
        splitter.setSizes([270, 910])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(splitter)
        self.layer_list.layerSelected.connect(self.layerSelected)
        self.layer_list.orderChanged.connect(self.orderChanged)
        self.layer_list.layerActionRequested.connect(self.layerActionRequested)
        self.layer_list.layerPropertiesRequested.connect(self.layerPropertiesRequested)
        self.add_layer_button.clicked.connect(self.addLayerRequested)
        self.graph.nodeActivated.connect(self._node_activated)
        self.graph.nodeActionRequested.connect(self._node_action)
        self.graph.nodePropertiesRequested.connect(self._node_properties)

    def set_layers(self, layers: list[LayerListItem], selected_id: str = "") -> None:
        self.layer_list.set_layers(layers, selected_id)

    def select_layer(self, layer_id: str) -> None:
        self.layer_list.select_layer(layer_id)

    def set_pipeline(self, snapshot: LayerPipelineSnapshot) -> None:
        self.graph.set_snapshot(snapshot)

    def _node_activated(self, node: PipelineNode) -> None:
        if node.representation_id:
            self.representationActivated.emit(node.representation_id)

    def _node_action(self, node: PipelineNode, action: str) -> None:
        snapshot = self.graph._snapshot
        if snapshot is not None:
            self.nodeActionRequested.emit(snapshot.layer_id, node, action)

    def _node_properties(self, node: PipelineNode) -> None:
        snapshot = self.graph._snapshot
        if snapshot is not None:
            self.nodePropertiesRequested.emit(snapshot.layer_id, node)

    def hideEvent(self, event) -> None:  # type: ignore[override]
        self.graph.save_layout()
        super().hideEvent(event)


__all__ = [
    "LayerManagerDialog",
    "LayerPipelineSnapshot",
    "ObjectHistoryEntry",
    "ObjectPropertiesDialog",
    "ObjectPropertiesSnapshot",
    "PipelineLane",
    "PipelineNode",
]
