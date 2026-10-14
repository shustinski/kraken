"""remote_client: remote history event."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True, slots=True)
class _RemoteHistoryEvent:
    _payload: Mapping[str, object]
    project_id: str = ""

    @property
    def event_id(self) -> str:
        return str(self._payload.get("event_id", ""))

    @property
    def event_type(self) -> str:
        return str(self._payload.get("event_type", ""))

    @property
    def stream_id(self) -> str:
        return str(self._payload.get("stream_id", ""))

    @property
    def revision(self) -> int:
        return int(self._payload.get("revision", 0))

    @property
    def recorded_at(self) -> datetime:
        value = self._payload.get("recorded_at")
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value))

    @property
    def payload(self) -> Mapping[str, object]:
        raw = self._payload.get("payload", {})
        return raw if isinstance(raw, Mapping) else {}

    @property
    def performer_id(self) -> str | None:
        value = self._payload.get("performer_id")
        return None if value is None else str(value)

    @property
    def actor(self) -> object:
        raw = self._payload.get("actor", {})
        if not isinstance(raw, Mapping):
            return None

        @dataclass(frozen=True, slots=True)
        class _Actor:
            display_name: str
            principal_id: str

        return _Actor(
            display_name=str(raw.get("display_name", "")),
            principal_id=str(raw.get("principal_id", "")),
        )
