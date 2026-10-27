"""Read-only history view for generic events without a domain actor envelope."""
from dataclasses import dataclass
from datetime import datetime
from collections.abc import Mapping

from kraken_manager.domain.events import ActorSnapshot, ProgramSnapshot
from kraken_manager.domain.identity import PrincipalProvider

from .event_store import CorruptEventLogError, StoredEvent


@dataclass(frozen=True, slots=True)
class GenericHistoryEvent:
    event_id: str
    project_id: str
    stream_id: str
    revision: int
    event_type: str
    payload: Mapping
    recorded_at: datetime
    idempotency_key: str | None = None
    # Generic analysis events did not persist a principal snapshot. Keep that
    # absence explicit instead of attributing them to the current user.
    actor: ActorSnapshot | None = None
    program: ProgramSnapshot | None = None
    performer_id: str | None = None
    effective_at: datetime | None = None

    @classmethod
    def from_stored(cls, stored: StoredEvent):
        data = stored.data
        try:
            kind, payload = data["event_type"], data["payload"]
            if not isinstance(kind, str) or not kind or not isinstance(payload, Mapping):
                raise ValueError("invalid event type or payload")
            timestamp = datetime.fromisoformat(data["recorded_at"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError("timestamp has no timezone")
            actor_value = data.get("actor")
            actor = None if actor_value is None else ActorSnapshot(
                principal_id=actor_value["principal_id"], provider=PrincipalProvider(actor_value["provider"]),
                subject=str(actor_value["subject"]), display_name=str(actor_value["display_name"]))
            program_value = data.get("program")
            program = None if program_value is None else ProgramSnapshot(name=program_value["name"], version=program_value.get("version"))
            effective = data.get("effective_at")
            effective = None if effective is None else datetime.fromisoformat(effective.replace("Z", "+00:00"))
            if effective is not None and effective.tzinfo is None:
                raise ValueError("effective timestamp has no timezone")
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise CorruptEventLogError(f"event {stored.event_id!r} cannot be displayed in project history: {exc}") from exc
        return cls(stored.event_id, stored.project_id, stored.stream_id, stored.revision,
            kind, payload, timestamp, data.get("idempotency_key"), actor, program, data.get("performer_id"), effective)
