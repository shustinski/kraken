"""Application notification and connection values shared by frontends and adapters."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProjectEventWake:
    """Compact wake signal from shared storage (WebSocket / outbox)."""

    project_id: str
    event_type: str
    event_id: str
    position: int | None = None
    revision: int | None = None
    stream_id: str = ""
    entity_kind: str = ""
    entity_id: str = ""


@dataclass(frozen=True, slots=True)
class ServerConnectionSummary:
    base_url: str
    account_subject: str
