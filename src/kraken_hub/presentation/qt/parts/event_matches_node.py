"""manager_app: event matches node."""
from __future__ import annotations
import os
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal
from kraken_manager.domain.file_change_history import events_for_file_change_history
from kraken_manager.presentation.qt import (
    ObjectHistoryEntry,
    PipelineNode,
)

class _RemoteSignalBridge(QObject):
    """Marshal transport-thread callbacks onto the Qt event loop."""

    wakeReceived = pyqtSignal(object)
    statusChanged = pyqtSignal(str)

def _plain_value(value):
    if is_dataclass(value):
        return _plain_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_plain_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value

def _is_network_path(path: Path | str) -> bool:
    raw = str(path)
    if raw.startswith(("\\\\", "//")):
        return True
    if os.name != "nt":
        return False
    try:
        import ctypes

        anchor = Path(raw).anchor
        return bool(anchor) and ctypes.windll.kernel32.GetDriveTypeW(anchor) == 4
    except (AttributeError, OSError):
        return False

def _confirm_server_external_path(parent, service, project_id: object, path: Path | str) -> bool:
    if not bool(getattr(service, "is_remote_project", lambda _value: False)(project_id)):
        return True
    if _is_network_path(path):
        return True
    from PyQt6.QtWidgets import QMessageBox

    answer = QMessageBox.warning(
        parent,
        "Локальный файл в серверном проекте",
        "Выбран файл с локального диска. Ссылка сохранится, но файл будет недоступен "
        "другим пользователям и серверным агентам. Продолжить?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel,
    )
    return answer == QMessageBox.StandardButton.Yes

def _event_payload(event) -> dict[str, object]:
    payload = getattr(event, "payload", {})
    return _plain_value(payload) if isinstance(payload, Mapping) else {}

def _nested_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key, {})
    return value if isinstance(value, Mapping) else {}

def _event_belongs_to_layer(event, layer_id: str) -> bool:
    identifier = str(layer_id)
    payload = _event_payload(event)
    candidates = {
        str(payload.get("layer_id", "")),
        str(_nested_mapping(payload, "layer").get("id", "")),
        str(_nested_mapping(payload, "job").get("layer_id", "")),
        str(_nested_mapping(payload, "manifest").get("layer_id", "")),
    }
    if identifier in candidates:
        return True
    if identifier in {str(value) for value in payload.get("layer_ids", ())}:
        return True
    stream_id = str(getattr(event, "stream_id", ""))
    return stream_id in {
        f"layer:{identifier}",
        f"layer-files:{identifier}",
        f"layer-pipeline:{identifier}",
        f"karakal:{identifier}",
    }

def _node_identifiers(node: PipelineNode) -> set[str]:
    identifiers = {str(node.node_id), str(node.representation_id or "")}
    for key in ("job_id", "run_id", "pipeline_event_id"):
        value = node.details.get(key)
        if value:
            identifiers.add(str(value))
    for prefix in ("action:", "workspace-job:", "workspace-output:", "karakal:"):
        if str(node.node_id).startswith(prefix):
            identifiers.add(str(node.node_id)[len(prefix):])
    if ":" in str(node.node_id):
        identifiers.add(str(node.node_id).split(":", 1)[0])
    return {value for value in identifiers if value}

def _event_matches_node(event, node: PipelineNode) -> bool:
    identifiers = _node_identifiers(node)
    if not identifiers or node.kind in {"missing", "blackbox"}:
        return False
    payload = _event_payload(event)
    manifest = _nested_mapping(payload, "manifest")
    parameters = _nested_mapping(manifest, "parameters")
    action_parameters = _nested_mapping(payload, "parameters")
    job = _nested_mapping(payload, "job")
    representation = _nested_mapping(payload, "representation")
    candidates = {
        str(getattr(event, "event_id", "")),
        str(payload.get("representation_id", "")),
        str(payload.get("plugin_job_id", "")),
        str(payload.get("run_id", "")),
        str(payload.get("action_event_id", "")),
        str(payload.get("node_id", "")),
        str(job.get("id", "")),
        str(job.get("target_representation_id", "")),
        str(manifest.get("target_representation_id", "")),
        str(manifest.get("source_representation_id", "")),
        str(parameters.get("source_representation_id", "")),
        str(action_parameters.get("source_representation_id", "")),
        str(payload.get("source_image_representation_id", "")),
        str(representation.get("id", "")),
        str(representation.get("source_image_representation_id", "")),
    }
    deactivated = {
        str(value) for value in payload.get("deactivated_representation_ids", ())
    }
    for value in payload.get("deactivated", ()):
        if isinstance(value, Mapping):
            deactivated.add(str(value.get("id", "")))
    if identifiers.intersection(candidates | deactivated):
        return True
    stream_id = str(getattr(event, "stream_id", ""))
    return any(stream_id.endswith(f":{identifier}") for identifier in identifiers)

def _history_entries(events) -> tuple[ObjectHistoryEntry, ...]:
    ordered = sorted(
        events_for_file_change_history(events),
        key=lambda event: (
            getattr(event, "recorded_at", None),
            str(getattr(event, "event_id", "")),
        ),
        reverse=True,
    )
    return tuple(
        ObjectHistoryEntry(
            recorded_at=event.recorded_at.isoformat(),
            actor=str(getattr(getattr(event, "actor", None), "display_name", "") or "—"),
            event_type=str(getattr(event, "event_type", "")),
            payload=_event_payload(event),
        )
        for event in ordered
    )

def _creator(events) -> tuple[str, str]:
    ordered = sorted(
        events,
        key=lambda event: (
            getattr(event, "recorded_at", None),
            str(getattr(event, "event_id", "")),
        ),
    )
    if not ordered:
        return "—", "—"
    event = ordered[0]
    actor = str(getattr(getattr(event, "actor", None), "display_name", "") or "—")
    recorded_at = getattr(event, "recorded_at", None)
    return actor, "—" if recorded_at is None else recorded_at.isoformat()

def _count_regular_files(paths) -> int | None:
    candidates = [Path(str(value)) for value in paths if str(value or "").strip()]
    if not candidates:
        return None
    seen: set[str] = set()
    accessible = False
    for candidate in candidates:
        try:
            if candidate.is_file():
                accessible = True
                seen.add(str(candidate.resolve()).casefold())
                continue
            if not candidate.is_dir():
                continue
            accessible = True
            for path in candidate.rglob("*"):
                try:
                    if path.is_file():
                        seen.add(str(path.resolve()).casefold())
                except OSError:
                    continue
        except OSError:
            continue
    return len(seen) if accessible else None
