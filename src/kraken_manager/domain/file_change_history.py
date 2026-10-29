"""File-change history shown for interactive edits and batch work."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

FILE_SCOPED_EVENT_TYPES = frozenset(
    {
        "ArtifactSeriesCreated",
        "ArtifactVersionCreated",
        "ArtifactVersionActivated",
    }
)
BATCH_LEVEL_EVENT_TYPES = frozenset(
    {
        "PluginResultImported",
        "ReviewReturnCommitted",
        "ReviewBatchAccepted",
        "LayerFilesImported",
    }
)
BATCH_FILE_HISTORY = "batch"
BATCH_FILE_CHANGE = "BatchFileChange"


@dataclass(frozen=True, slots=True)
class BatchFileHistoryEvent:
    """One history line for a batch of files."""

    event_id: str
    recorded_at: datetime
    event_type: str
    payload: Mapping[str, object]
    actor: object


def batch_file_history_fields(batch_key: str) -> dict[str, str]:
    key = str(batch_key).strip()
    if not key:
        raise ValueError("batch file history requires a batch key")
    return {"file_history": BATCH_FILE_HISTORY, "batch_key": key}


def _payload(event: object) -> Mapping[str, object]:
    payload = getattr(event, "payload", {})
    return payload if isinstance(payload, Mapping) else {}


def _batch_level_keys(event: object) -> set[str]:
    if str(getattr(event, "event_type", "")) not in BATCH_LEVEL_EVENT_TYPES:
        return set()
    payload = _payload(event)
    keys: set[str] = set()
    for field_name in ("batch_key", "plugin_job_id", "review_batch_id"):
        value = payload.get(field_name)
        if value:
            keys.add(str(value))
    job = payload.get("job")
    if isinstance(job, Mapping) and job.get("id"):
        keys.add(str(job["id"]))
    return keys


def _group_key(event: object) -> str:
    payload = _payload(event)
    for field_name in ("batch_key", "plugin_job_id", "review_batch_id"):
        value = payload.get(field_name)
        if value:
            return str(value)
    return str(getattr(event, "correlation_id", "") or "batch")


def _file_count(events: Sequence[object]) -> int:
    versions = [
        event
        for event in events
        if str(getattr(event, "event_type", "")) == "ArtifactVersionCreated"
    ]
    return len(versions) if versions else len(events)


def events_for_file_change_history(events: Sequence[object]) -> tuple[object, ...]:
    """Drop per-file batch entries. Keep one batch line and every interactive edit."""

    batch_level_keys: set[str] = set()
    omitted: dict[str, list[object]] = {}
    visible: list[object] = []
    for event in events:
        batch_level_keys.update(_batch_level_keys(event))
        payload = _payload(event)
        if (
            str(getattr(event, "event_type", "")) in FILE_SCOPED_EVENT_TYPES
            and payload.get("file_history") == BATCH_FILE_HISTORY
        ):
            omitted.setdefault(_group_key(event), []).append(event)
            continue
        visible.append(event)
    for key, group in omitted.items():
        if key in batch_level_keys or not group:
            continue
        latest = max(group, key=lambda event: getattr(event, "recorded_at", None))
        actor = getattr(latest, "actor", None)
        visible.append(
            BatchFileHistoryEvent(
                event_id=f"batch-file-change:{key}",
                recorded_at=latest.recorded_at,
                event_type=BATCH_FILE_CHANGE,
                payload={
                    "batch_key": key,
                    "file_count": _file_count(group),
                    "file_history": BATCH_FILE_HISTORY,
                },
                actor=actor,
            )
        )
    return tuple(visible)
