"""layer_management: format scalar."""
from __future__ import annotations
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from kraken_manager.presentation.qt.layer_management_parts.module_state import (
    _EVENT_TYPE_LABELS,
    _FIELD_LABELS,
    _ID_FIELD_SUFFIXES,
    _ROLE_LABELS,
    _STATE_LABELS,
)

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
