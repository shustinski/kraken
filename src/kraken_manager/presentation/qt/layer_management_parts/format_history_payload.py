"""layer_management: format history payload."""
from __future__ import annotations
from collections.abc import Mapping

from kraken_manager.presentation.qt.layer_management_parts.format_scalar import (
    _append_lines,
    _append_mapping_fields,
    _is_uuid_like,
    _should_skip_key,
)

from kraken_manager.presentation.qt.layer_management_parts.selection_summary import _selection_summary

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
