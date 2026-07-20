from __future__ import annotations

import dataclasses
import json
import re
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from ._atomic import atomic_write_bytes
from .layout import FileProjectLayout
from .locking import ProjectFileLock


_SEGMENT_NAME = re.compile(r"^(?P<first>[0-9]{20})-(?P<last>[0-9]{20})\.jsonl$")


class EventStreamConflict(RuntimeError):
    def __init__(self, stream_id: str, expected_revision: int, actual_revision: int) -> None:
        super().__init__(
            f"stream {stream_id!r} revision conflict: expected {expected_revision}, actual {actual_revision}"
        )
        self.stream_id = stream_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision


class CorruptEventLogError(RuntimeError):
    pass


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _json_compatible(value.value)
    if isinstance(value, (datetime, date)):
        if isinstance(value, datetime) and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_compatible(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_compatible(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_compatible(to_dict())
    raise TypeError(f"event value {type(value).__name__} is not JSON serializable")


def _default_encode(event: Any) -> dict[str, Any]:
    encoded = _json_compatible(event)
    if not isinstance(encoded, dict):
        raise TypeError("an event must encode to a JSON object")
    return encoded


def _validate_stream_id(stream_id: str) -> str:
    if not isinstance(stream_id, str) or not stream_id.strip() or len(stream_id) > 512 or "\x00" in stream_id:
        raise ValueError("stream_id must be a non-empty string of at most 512 characters")
    return stream_id


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError("event timestamp must be an ISO-8601 string")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclasses.dataclass(frozen=True, slots=True)
class StoredEvent(Mapping[str, Any]):
    data: Mapping[str, Any]
    global_position: int

    @property
    def project_id(self) -> str:
        return str(self.data["project_id"])

    @property
    def stream_id(self) -> str:
        return str(self.data["stream_id"])

    @property
    def revision(self) -> int:
        return int(self.data["revision"])

    @property
    def event_id(self) -> str:
        return str(self.data["event_id"])

    def to_dict(self, *, include_storage_metadata: bool = False) -> dict[str, Any]:
        value = dict(self.data)
        if include_storage_metadata:
            value["_storage"] = {"global_position": self.global_position}
        return value

    def __getitem__(self, key: str) -> Any:
        if key == "global_position":
            return self.global_position
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.data

    def __len__(self) -> int:
        return len(self.data)


def decode_event_envelope(stored: StoredEvent) -> Any:
    """Decode canonical domain envelopes while retaining generic-event support."""

    required = {"event_type", "payload", "schema_version", "recorded_at", "actor"}
    if not required.issubset(stored.data):
        return stored

    from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
    from kraken_manager.domain.identity import PrincipalProvider

    value = dict(stored.data)
    actor_value = value["actor"]
    if not isinstance(actor_value, Mapping):
        raise CorruptEventLogError(f"event {stored.event_id!r} has an invalid actor snapshot")
    actor = ActorSnapshot(
        principal_id=actor_value["principal_id"],
        provider=PrincipalProvider(actor_value["provider"]),
        subject=str(actor_value["subject"]),
        display_name=str(actor_value["display_name"]),
    )
    program_value = value.get("program")
    program = None
    if isinstance(program_value, Mapping):
        program = ProgramSnapshot(name=str(program_value["name"]), version=program_value.get("version"))
    return EventEnvelope(
        event_id=str(value["event_id"]),
        stream_id=str(value["stream_id"]),
        project_id=value["project_id"],
        revision=int(value["revision"]),
        event_type=str(value["event_type"]),
        payload=value["payload"],
        schema_version=int(value["schema_version"]),
        recorded_at=_parse_datetime(value["recorded_at"]),
        actor=actor,
        effective_at=(None if value.get("effective_at") is None else _parse_datetime(value["effective_at"])),
        performer_id=value.get("performer_id"),
        program=program,
        correlation_id=value.get("correlation_id"),
        causation_id=value.get("causation_id"),
        idempotency_key=value.get("idempotency_key"),
    )


class FilesystemEventStore:
    """Project-scoped append-only JSONL event store.

    Every append creates a new immutable segment through an atomic rename.  Thus a
    multi-event append is visible in full or not at all after a process or machine
    failure.  The event log, not the optional SQLite index, is authoritative.
    """

    def __init__(
        self,
        catalog_root: str | Path,
        project_id: str,
        *,
        encoder: Callable[[Any], Mapping[str, Any]] | None = None,
        decoder: Callable[[StoredEvent], Any] | None = None,
        lock_timeout: float | None = 10.0,
    ) -> None:
        self.layout = FileProjectLayout(Path(catalog_root), project_id)
        self.layout.ensure_directories()
        self.project_id = project_id
        self._encode = encoder or _default_encode
        self._decode = decoder or decode_event_envelope
        self.lock_timeout = lock_timeout
        self.lock = ProjectFileLock(self.layout.lock_path)

    def append(
        self,
        stream_id: str,
        *,
        expected_revision: int,
        events: Sequence[Any] | Iterable[Any],
    ) -> int:
        _validate_stream_id(stream_id)
        if expected_revision < 0:
            raise ValueError("expected_revision must be zero or greater")
        batch = list(events)

        with self.lock.hold(self.lock_timeout):
            existing = list(self._iter_stored_unlocked())
            actual_revision = max(
                (event.revision for event in existing if event.stream_id == stream_id),
                default=0,
            )
            if actual_revision != expected_revision:
                raise EventStreamConflict(stream_id, expected_revision, actual_revision)
            if not batch:
                return actual_revision

            known_event_ids = {event.event_id for event in existing}
            next_position = existing[-1].global_position + 1 if existing else 1
            records: list[dict[str, Any]] = []
            for offset, event in enumerate(batch, start=1):
                value = dict(self._encode(event))
                if "_storage" in value:
                    raise ValueError("event field '_storage' is reserved")
                self._assert_or_set(value, "project_id", self.project_id)
                self._assert_or_set(value, "stream_id", stream_id)
                self._assert_or_set(value, "revision", expected_revision + offset)
                event_id = str(value.get("event_id") or uuid.uuid4())
                if event_id in known_event_ids:
                    raise ValueError(f"duplicate event_id {event_id!r}")
                known_event_ids.add(event_id)
                value["event_id"] = event_id
                position = next_position + offset - 1
                value["_storage"] = {"global_position": position}
                records.append(value)

            self._write_segment(records, next_position)
            return expected_revision + len(records)

    def append_preserved(self, events: Sequence[Any] | Iterable[Any]) -> int:
        """Append migrated envelopes while preserving their IDs and revisions.

        This method is intentionally stricter than :meth:`append`: every envelope
        must already contain its identity and may carry the canonical ``_storage``
        position.  A whole import chunk is published atomically.
        """

        batch = list(events)
        if not batch:
            return self.last_global_position()
        with self.lock.hold(self.lock_timeout):
            existing = list(self._iter_stored_unlocked())
            next_position = existing[-1].global_position + 1 if existing else 1
            known_event_ids = {event.event_id for event in existing}
            stream_revisions: dict[str, int] = {}
            for event in existing:
                stream_revisions[event.stream_id] = event.revision

            records: list[dict[str, Any]] = []
            for offset, event in enumerate(batch):
                value = dict(self._encode(event))
                storage = value.pop("_storage", None)
                position = next_position + offset
                if storage is not None:
                    supplied = storage.get("global_position") if isinstance(storage, Mapping) else None
                    if supplied != position:
                        raise ValueError(
                            f"migrated event global position mismatch: expected {position}, found {supplied}"
                        )
                if value.get("project_id") != self.project_id:
                    raise ValueError("migrated event belongs to another project")
                try:
                    stream_id = str(value["stream_id"])
                    revision = int(value["revision"])
                    event_id = str(value["event_id"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("migrated event has an incomplete envelope") from exc
                _validate_stream_id(stream_id)
                expected_revision = stream_revisions.get(stream_id, 0) + 1
                if revision != expected_revision:
                    raise EventStreamConflict(stream_id, expected_revision - 1, revision - 1)
                if event_id in known_event_ids:
                    raise ValueError(f"duplicate event_id {event_id!r}")
                known_event_ids.add(event_id)
                stream_revisions[stream_id] = revision
                value["_storage"] = {"global_position": position}
                records.append(value)

            self._write_segment(records, next_position)
            return next_position + len(records) - 1

    def _write_segment(self, records: Sequence[Mapping[str, Any]], first_position: int) -> None:
        last_position = first_position + len(records) - 1
        segment_path = self.layout.events_dir / f"{first_position:020d}-{last_position:020d}.jsonl"
        payload = b"".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
            for record in records
        )
        try:
            atomic_write_bytes(segment_path, payload, overwrite=False)
        except FileExistsError as exc:
            raise CorruptEventLogError(f"event segment already exists: {segment_path.name}") from exc

    @staticmethod
    def _assert_or_set(value: dict[str, Any], field: str, expected: Any) -> None:
        if field in value and value[field] != expected:
            raise ValueError(f"event {field} {value[field]!r} does not match {expected!r}")
        value[field] = expected

    def load_stream(
        self,
        stream_id: str,
        *,
        after_revision: int = 0,
        as_of: datetime | None = None,
    ) -> tuple[Any, ...]:
        _validate_stream_id(stream_id)
        result: list[Any] = []
        for event in self.iter_project():
            if event.stream_id != stream_id or event.revision <= after_revision:
                continue
            if as_of is not None:
                recorded_at = event.data.get("recorded_at")
                if recorded_at is None or _parse_datetime(recorded_at) > as_of:
                    continue
            result.append(self._decode(event))
        return tuple(result)

    read_stream = load_stream

    def iter_project(self, *, after_global_position: int = 0) -> Iterator[StoredEvent]:
        for event in self._iter_stored_unlocked():
            if event.global_position > after_global_position:
                yield event

    def current_revision(self, stream_id: str) -> int:
        _validate_stream_id(stream_id)
        return max(
            (event.revision for event in self._iter_stored_unlocked() if event.stream_id == stream_id),
            default=0,
        )

    def find_by_idempotency_key(self, project_id: Any, idempotency_key: str) -> tuple[Any, ...]:
        if project_id is not None and str(project_id) != self.project_id:
            return ()
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotency_key must be non-empty")
        return tuple(
            self._decode(event)
            for event in self.iter_project()
            if event.data.get("idempotency_key") == idempotency_key
        )

    def last_global_position(self) -> int:
        last = 0
        for event in self._iter_stored_unlocked():
            last = event.global_position
        return last

    def _segment_paths(self) -> list[tuple[int, int, Path]]:
        result: list[tuple[int, int, Path]] = []
        for path in self.layout.events_dir.glob("*.jsonl"):
            match = _SEGMENT_NAME.fullmatch(path.name)
            if match is None:
                raise CorruptEventLogError(f"unrecognized event segment name: {path.name}")
            first = int(match.group("first"))
            last = int(match.group("last"))
            if last < first:
                raise CorruptEventLogError(f"invalid event segment range: {path.name}")
            result.append((first, last, path))
        result.sort(key=lambda item: item[0])
        return result

    def _iter_stored_unlocked(self) -> Iterator[StoredEvent]:
        expected_position = 1
        seen_event_ids: set[str] = set()
        stream_revisions: dict[str, int] = {}

        for first, last, path in self._segment_paths():
            if first != expected_position:
                raise CorruptEventLogError(
                    f"event position gap or overlap before {path.name}: expected {expected_position}, found {first}"
                )
            count = 0
            try:
                stream = path.open("r", encoding="utf-8", newline="")
                with stream:
                    for line_number, line in enumerate(stream, start=1):
                        if not line.endswith("\n"):
                            raise CorruptEventLogError(f"truncated final record in {path.name}:{line_number}")
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise CorruptEventLogError(f"invalid JSON in {path.name}:{line_number}") from exc
                        if not isinstance(record, dict):
                            raise CorruptEventLogError(f"event record is not an object in {path.name}:{line_number}")
                        storage = record.pop("_storage", None)
                        position = storage.get("global_position") if isinstance(storage, dict) else None
                        if position != expected_position:
                            raise CorruptEventLogError(
                                f"invalid global position in {path.name}:{line_number}; expected {expected_position}"
                            )
                        try:
                            project_id = str(record["project_id"])
                            stream_id = str(record["stream_id"])
                            revision = int(record["revision"])
                            event_id = str(record["event_id"])
                        except (KeyError, TypeError, ValueError) as exc:
                            raise CorruptEventLogError(f"incomplete event envelope in {path.name}:{line_number}") from exc
                        if project_id != self.project_id:
                            raise CorruptEventLogError(f"event belongs to project {project_id!r}, not {self.project_id!r}")
                        expected_revision = stream_revisions.get(stream_id, 0) + 1
                        if revision != expected_revision:
                            raise CorruptEventLogError(
                                f"stream {stream_id!r} revision gap: expected {expected_revision}, found {revision}"
                            )
                        if event_id in seen_event_ids:
                            raise CorruptEventLogError(f"duplicate event_id {event_id!r}")
                        seen_event_ids.add(event_id)
                        stream_revisions[stream_id] = revision
                        yield StoredEvent(record, expected_position)
                        expected_position += 1
                        count += 1
            except UnicodeDecodeError as exc:
                raise CorruptEventLogError(f"event segment is not valid UTF-8: {path.name}") from exc
            if count != last - first + 1 or expected_position - 1 != last:
                raise CorruptEventLogError(f"event count does not match segment range in {path.name}")

    def verify(self) -> tuple[int, int]:
        count = 0
        streams: set[str] = set()
        for event in self._iter_stored_unlocked():
            count += 1
            streams.add(event.stream_id)
        return count, len(streams)
